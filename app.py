#!/usr/bin/env python3
"""
Main application for RAG + SQL hybrid query system
Terminal-based interface without Streamlit
"""

import os
import asyncio
import pandas as pd
from sqlalchemy import create_engine, text
import nest_asyncio

# Import our modules
from tools import setup_document_tool, setup_sql_tool
from workflow import RouterOutputAgentWorkflow

# Apply nest_asyncio for asyncio compatibility
nest_asyncio.apply()

# LlamaIndex imports
from llama_index.core import Settings
from llama_index.llms.openrouter import OpenRouter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def initialize_models():
    """Initialize LLM and embedding models"""
    print("Initializing models...")
    
    # Get API key from environment or user input
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        api_key = input("Enter your OpenRouter API key: ").strip()
        if not api_key:
            raise ValueError("OpenRouter API key is required")
    
    try:
        # Initialize LLM and embedding model - using a more capable model for better tool selection
        llm = OpenRouter(model="anthropic/claude-3.5-sonnet", api_key=api_key)
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        
        # Set global settings
        Settings.llm = llm
        Settings.embed_model = embed_model
        
        print("✓ Models initialized successfully!")
        return llm, embed_model
    except Exception as e:
        print(f"✗ Error initializing models: {e}")
        raise


def excel_to_sqlite(excel_path, db_path="database.sqlite", table_name="data"):
    """Convert Excel file to SQLite database"""
    print(f"Converting Excel file: {excel_path}")
    
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Excel file not found: {excel_path}")
    
    try:
        # Read Excel file
        df = pd.read_excel(excel_path)
        print(f"✓ Loaded Excel with {len(df)} rows and {len(df.columns)} columns")
        
        # Display column info
        print("Columns found:")
        for i, col in enumerate(df.columns):
            print(f"  {i+1}. {col} ({df[col].dtype})")
        
        # Create SQLite database
        engine = create_engine(f"sqlite:///{db_path}")
        df.to_sql(table_name, engine, if_exists='replace', index=False)
        
        print(f"✓ Database created: {db_path}")
        print(f"✓ Table created: {table_name}")
        
        return db_path, table_name, df.columns.tolist()
        
    except Exception as e:
        print(f"✗ Error converting Excel to SQLite: {e}")
        raise


def get_document_path():
    """Get document directory path from user"""
    while True:
        doc_path = input("Enter the path to your documents directory: ").strip()
        if not doc_path:
            print("✗ Document path is required")
            continue
        
        if not os.path.exists(doc_path):
            print(f"✗ Directory not found: {doc_path}")
            continue
        
        if not os.path.isdir(doc_path):
            print(f"✗ Path is not a directory: {doc_path}")
            continue
        
        # Check for supported files
        supported_files = []
        for file in os.listdir(doc_path):
            if file.lower().endswith(('.pdf', '.docx', '.pptx', '.txt')):
                supported_files.append(file)
        
        if not supported_files:
            print(f"✗ No supported files found in {doc_path}")
            print("Supported formats: PDF, DOCX, PPTX, TXT")
            continue
        
        print(f"✓ Found {len(supported_files)} supported files:")
        for file in supported_files[:5]:  # Show first 5 files
            print(f"  - {file}")
        if len(supported_files) > 5:
            print(f"  ... and {len(supported_files) - 5} more files")
        
        return doc_path


async def process_query(query, workflow):
    """Process a query using the workflow"""
    try:
        print(f"\n🔍 Processing: {query}")
        print("⏳ Please wait...")
        
        # Don't clear chat history - keep system context
        # Only clear if we have too many messages to avoid token limits
        if len(workflow.chat_history) > 20:
            # Keep system message and clear the rest
            system_msg = workflow.chat_history[0] if workflow.chat_history and workflow.chat_history[0].role == "system" else None
            workflow.chat_history = [system_msg] if system_msg else []
        
        # Run the workflow
        result = await asyncio.wait_for(workflow.run(message=query), timeout=60.0)
        
        # Extract tool usage information
        tools_used = []
        tool_names_seen = set()
        for msg in workflow.chat_history:
            if msg.role == "tool" and hasattr(msg, 'additional_kwargs'):
                tool_name = msg.additional_kwargs.get('tool_used', 'Unknown')
                if "cancelled" not in msg.content.lower() and "error" not in msg.content.lower():
                    if tool_name not in tool_names_seen:
                        tools_used.append({
                            'name': tool_name,
                            'response': msg.content
                        })
                        tool_names_seen.add(tool_name)
        
        # Format response
        if tools_used:
            unique_tool_names = [t['name'] for t in tools_used]
            print(f"\n🔧 Tools used: {', '.join(unique_tool_names)}")
            print(f"📝 Response:\n{result}")
        else:
            print(f"📝 Response:\n{result}")
            
        return result
        
    except asyncio.TimeoutError:
        error_msg = "⏰ Query timed out. Please try a simpler question."
        print(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"❌ Error processing query: {str(e)}"
        print(error_msg)
        return error_msg


def display_database_info(db_path, table_name):
    """Display information about the database"""
    try:
        engine = create_engine(f"sqlite:///{db_path}")
        
        # Get table info
        with engine.connect() as conn:
            # Get row count
            result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
            row_count = result.scalar()
            
            # Get column info
            result = conn.execute(text(f"PRAGMA table_info({table_name})"))
            columns = result.fetchall()
            
            print(f"\n📊 Database Information:")
            print(f"  Database: {db_path}")
            print(f"  Table: {table_name}")
            print(f"  Rows: {row_count:,}")
            print(f"  Columns: {len(columns)}")
            
            print(f"\n📋 Column Schema:")
            for col in columns:
                print(f"  - {col[1]} ({col[2]})")
                
            # Show sample data
            sample_df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 3", engine)
            print(f"\n📄 Sample Data:")
            print(sample_df.to_string(index=False))
            
    except Exception as e:
        print(f"❌ Error displaying database info: {e}")


def main():
    """Main application function"""
    print("🚀 RAG + SQL Hybrid Query System")
    print("=" * 50)
    
    try:
        # Initialize models
        llm, embed_model = initialize_models()
        
        # Get Excel file path and convert to SQLite
        while True:
            excel_path = input("\nEnter the path to your Excel file: ").strip()
            if not excel_path:
                print("✗ Excel file path is required")
                continue
            
            if not os.path.exists(excel_path):
                print(f"✗ File not found: {excel_path}")
                continue
            
            if not excel_path.lower().endswith(('.xlsx', '.xls')):
                print("✗ File must be an Excel file (.xlsx or .xls)")
                continue
            
            break
        
        # Convert Excel to SQLite
        db_path, table_name, columns = excel_to_sqlite(excel_path)
        
        # Get document directory
        doc_path = get_document_path()
        
        # Setup tools
        print("\n🔧 Setting up tools...")
        sql_tool = setup_sql_tool(db_path, table_name)
        doc_tool = setup_document_tool(doc_path)
        tools = [sql_tool, doc_tool]
        print("✓ Tools ready!")
        
        # Initialize workflow
        print("🔄 Initializing workflow...")
        workflow = RouterOutputAgentWorkflow(tools=tools, verbose=False, timeout=120)
        print("✓ Workflow ready!")
        
        # Display database info
        display_database_info(db_path, table_name)
        
        # Start interactive loop
        print(f"\n💬 Interactive Query Mode")
        print(f"Type your questions below. Use 'quit', 'exit', or 'q' to stop.")
        print(f"Examples:")
        print(f"  - Questions about your Excel data will use SQL")
        print(f"  - Questions about uploaded documents will use RAG")
        print("-" * 50)
        
        while True:
            try:
                query = input("\n❓ Your question: ").strip()
                
                if not query:
                    continue
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                # Process the query
                response = asyncio.run(process_query(query, workflow))
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
                continue
                
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())