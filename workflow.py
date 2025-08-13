"""
Router Agent Workflow for RAG + SQL hybrid system
Simplified version for terminal use
"""

import asyncio
from typing import Dict, List, Any, Optional
from llama_index.core import Settings
from llama_index.core.tools import BaseTool
from llama_index.core.llms import ChatMessage
from llama_index.core.llms.llm import ToolSelection, LLM
from llama_index.core.workflow import (
    Workflow,
    Event,
    StartEvent,
    StopEvent,
    step,
    Context,
)


#####################################
# Define Events
#####################################
class InputEvent(Event):
    """Input event for processing queries."""
    pass


class GatherToolsEvent(Event):
    """Event for gathering tool calls."""
    tool_calls: Any


class ToolCallEvent(Event):
    """Event for individual tool calls."""
    tool_call: ToolSelection


class ToolCallEventResult(Event):
    """Event for tool call results."""
    msg: ChatMessage


#####################################
# Router Workflow
#####################################
class RouterOutputAgentWorkflow(Workflow):
    """Router workflow that decides between SQL and document tools."""

    def __init__(
        self,
        tools: List[BaseTool],
        timeout: Optional[float] = 10.0,
        disable_validation: bool = False,
        verbose: bool = False,
        llm: Optional[LLM] = None,
        chat_history: Optional[List[ChatMessage]] = None,
    ):
        """Initialize the workflow."""
        super().__init__(
            timeout=timeout, 
            disable_validation=disable_validation, 
            verbose=verbose
        )
        self.tools: List[BaseTool] = tools
        self.tools_dict: Optional[Dict[str, BaseTool]] = {
            tool.metadata.name: tool for tool in self.tools
        }
        
        # Use provided LLM or fall back to Settings.llm
        self.llm: LLM = llm or Settings.llm
        if self.llm is None:
            raise ValueError("No LLM provided and Settings.llm is not initialized")
            
        self.chat_history: List[ChatMessage] = chat_history or []

    def reset(self) -> None:
        """Reset chat history."""
        self.chat_history = []

    @step()
    async def prepare_chat(self, ev: StartEvent) -> InputEvent:
        """Prepare the chat by adding user message to history."""
        message = ev.get("message")
        if message is None:
            raise ValueError("'message' field is required.")

        # Add system message if this is the first message (empty chat history)
        if not self.chat_history:
            system_msg = ChatMessage(
                role="system", 
                content=(
                    "You are an intelligent assistant with access to two specialized tools:\n\n"
                    "1. SQL Tool: For querying structured data about people/students including names, "
                    "birthdays, contact info, addresses, etc. Use this for specific person lookups "
                    "and data analysis.\n\n"
                    "2. Document Tool: For searching uploaded documents about policies, procedures, "
                    "institutional information, etc. Use this for conceptual questions.\n\n"
                    "IMPORTANT GUIDELINES:\n"
                    "- When a user asks about a specific person's information (name, birthday, phone, email, address), use ONLY the sql_tool\n"
                    "- When they ask about policies, procedures, or general information from documents, use ONLY the document_tool\n"
                    "- Make ONE targeted tool call per query - do not make multiple calls to the same tool\n"
                    "- Choose the most appropriate single tool based on the question type\n"
                    "- For mixed queries (e.g., 'terms and conditions AND phone owner'), make separate tool calls for each part"
                )
            )
            self.chat_history.append(system_msg)

        # Add user message to chat history
        self.chat_history.append(ChatMessage(role="user", content=message))
        return InputEvent()

    @step()
    async def chat(self, ev: InputEvent) -> GatherToolsEvent | StopEvent:
        """Process the chat and determine if tools are needed."""
        try:
            if self._verbose:
                print("💭 LLM is analyzing your query...")
            
            # Get response from LLM with tools
            chat_res = await self.llm.achat_with_tools(
                self.tools,
                chat_history=self.chat_history,
                verbose=self._verbose,
                allow_parallel_tool_calls=False,
            )
            
            # Extract tool calls
            tool_calls = self.llm.get_tool_calls_from_response(
                chat_res, error_on_no_tool_call=False
            )

            # Add AI message to history
            ai_message = chat_res.message
            self.chat_history.append(ai_message)
            
            if self._verbose:
                if tool_calls:
                    print(f"🔧 LLM decided to use {len(tool_calls)} tool(s)")
                else:
                    print("💬 LLM provided direct response")

            # If no tools needed, return direct response
            if not tool_calls:
                return StopEvent(result=ai_message.content)

            # Otherwise, proceed with tool calls
            return GatherToolsEvent(tool_calls=tool_calls)
            
        except asyncio.CancelledError:
            print("⚠️ Chat operation was cancelled")
            return StopEvent(result="The operation was cancelled. Please try again.")
        except Exception as e:
            error_msg = f"Error during chat: {str(e)}"
            if self._verbose:
                print(f"❌ {error_msg}")
            return StopEvent(
                result="I encountered an issue processing your request. Please try rephrasing your question."
            )

    @step(pass_context=True)
    async def dispatch_calls(self, ctx: Context, ev: GatherToolsEvent) -> ToolCallEvent:
        """Dispatch tool calls to be executed."""
        tool_calls = ev.tool_calls
        await ctx.set("num_tool_calls", len(tool_calls))

        # Send tool call events
        for tool_call in tool_calls:
            ctx.send_event(ToolCallEvent(tool_call=tool_call))

        return None

    @step()
    async def call_tool(self, ev: ToolCallEvent) -> ToolCallEventResult:
        """Execute a tool call."""
        try:
            tool_call = ev.tool_call
            tool_id = tool_call.tool_id

            if self._verbose:
                print(f"🔧 Executing {tool_call.tool_name}...")

            # Get the tool and call it
            tool = self.tools_dict[tool_call.tool_name]
            output = await tool.acall(**tool_call.tool_kwargs)
            
            # Prepare response content
            content = str(output) if output is not None else ""
            
            # Create response message
            msg = ChatMessage(
                name=tool_call.tool_name,
                content=content,
                role="tool",
                additional_kwargs={
                    "tool_call_id": tool_id, 
                    "name": tool_call.tool_name,
                    "tool_used": tool_call.tool_name
                },
            )

            if self._verbose:
                response_preview = content[:100] + "..." if len(content) > 100 else content
                print(f"✓ {tool_call.tool_name} completed: {response_preview}")

            return ToolCallEventResult(msg=msg)
            
        except asyncio.CancelledError:
            print(f"⚠️ Tool call {tool_call.tool_name} was cancelled")
            msg = ChatMessage(
                name=tool_call.tool_name,
                content="Tool execution was cancelled",
                role="tool",
                additional_kwargs={
                    "tool_call_id": tool_id, 
                    "name": tool_call.tool_name, 
                    "tool_used": tool_call.tool_name
                },
            )
            return ToolCallEventResult(msg=msg)
        except Exception as e:
            error_msg = f"Error executing {tool_call.tool_name}: {str(e)}"
            if self._verbose:
                print(f"❌ {error_msg}")
            msg = ChatMessage(
                name=tool_call.tool_name,
                content=error_msg,
                role="tool",
                additional_kwargs={
                    "tool_call_id": tool_id, 
                    "name": tool_call.tool_name, 
                    "tool_used": tool_call.tool_name
                },
            )
            return ToolCallEventResult(msg=msg)

    @step(pass_context=True)
    async def gather(self, ctx: Context, ev: ToolCallEventResult) -> StopEvent | None:
        """Gather all tool call results and continue or finish."""
        try:
            # Wait for all tool call events to complete
            tool_events = ctx.collect_events(
                ev, [ToolCallEventResult] * await ctx.get("num_tool_calls")
            )
            
            if not tool_events:
                return None

            # Add all tool responses to chat history
            for tool_event in tool_events:
                self.chat_history.append(tool_event.msg)

            if self._verbose:
                print("🔄 Processing tool results...")

            # Continue the conversation loop with tool results
            return InputEvent()
            
        except Exception as e:
            error_msg = f"Error gathering tool results: {str(e)}"
            if self._verbose:
                print(f"❌ {error_msg}")
            return StopEvent(
                result="I encountered an issue processing the tool responses. Please try again."
            )


def create_workflow(tools: List[BaseTool], verbose: bool = False) -> RouterOutputAgentWorkflow:
    """Create and return a configured workflow."""
    try:
        workflow = RouterOutputAgentWorkflow(
            tools=tools,
            verbose=verbose,
            timeout=120
        )
        return workflow
    except Exception as e:
        print(f"❌ Error creating workflow: {e}")
        raise


async def run_query(workflow: RouterOutputAgentWorkflow, query: str) -> str:
    """Run a single query through the workflow."""
    try:
        # Clear chat history for fresh query
        workflow.reset()
        
        # Run the workflow
        result = await workflow.run(message=query)
        return str(result)
        
    except Exception as e:
        error_msg = f"Error running query: {str(e)}"
        print(f"❌ {error_msg}")
        return error_msg