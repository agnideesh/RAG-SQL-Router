"""
Tool setup functions for RAG + SQL hybrid system
Simplified version without Codex and using simple vector store
"""

import os
import uuid
from sqlalchemy import create_engine
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    SQLDatabase,
    PromptTemplate,
    StorageContext,
)
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.core.tools import QueryEngineTool, FunctionTool
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.readers.docling import DoclingReader
from llama_index.core.vector_stores import SimpleVectorStore



def setup_sql_tool(db_path="database.sqlite", table_name="data"):
    """Setup SQL query tool for querying database."""
    # Validate database exists
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file not found: {db_path}")

    try:
        engine = create_engine(f"sqlite:///{db_path}")
        sql_database = SQLDatabase(engine)
        print(f"✓ Connected to database: {db_path}")
    except Exception as e:
        print(f"✗ Error setting up SQL database: {e}")
        raise

    # Get schema information
    schema_info = get_database_schema(db_path, table_name)
    schema_description = "Available columns:\n"
    for col in schema_info:
        schema_description += f"  - {col['name']} ({col['type']})\n"

    # Create SQL query engine
    sql_query_engine = NLSQLTableQueryEngine(
        sql_database=sql_database,
        tables=[table_name],
    )

    # Create tool for SQL querying with enhanced description
    sql_tool = QueryEngineTool.from_defaults(
        query_engine=sql_query_engine,
        name="sql_tool",
        description=(
            f"Use this tool to query structured data from the '{table_name}' table. "
            f"This contains student/person data with the following information:\n"
            f"{schema_description}\n"
            f"Use this tool for:\n"
            f"- Finding specific people by name (e.g., 'Paula Walker', 'John Smith')\n"
            f"- Getting personal details like birthdays, phone numbers, emails, addresses\n"
            f"- Filtering data by criteria (age, location, etc.)\n"
            f"- Statistical queries and data analysis\n"
            f"- Any questions about the structured data in the Excel file\n\n"
            f"Examples of queries for this tool:\n"
            f"- 'What is the birthday of Paula Walker?'\n"
            f"- 'Find all students from California'\n"
            f"- 'How many people are in the database?'"
        ),
    )

    print(f"✓ SQL tool created for table: {table_name}")
    return sql_tool


def setup_document_tool(file_dir):
    """Setup document query tool from directory with simple vector store."""
    if not os.path.exists(file_dir):
        raise FileNotFoundError(f"Document directory not found: {file_dir}")
    
    if not os.path.isdir(file_dir):
        raise ValueError(f"Path is not a directory: {file_dir}")

    print(f"📄 Processing documents from: {file_dir}")

    # Create readers and parsers
    reader = DoclingReader()
    node_parser = MarkdownNodeParser()
    
    # Load documents
    try:
        # Use SimpleDirectoryReader without custom extractors for .txt files
        # It will use the default text reader for .txt files
        loader = SimpleDirectoryReader(
            input_dir=file_dir,
            file_extractor={
                ".pdf": reader,
                ".docx": reader,
                ".pptx": reader,
                # Don't specify .txt - let SimpleDirectoryReader handle it with default reader
            },
        )
        docs = loader.load_data()
        print(f"✓ Loaded {len(docs)} documents")
    except Exception as e:
        print(f"✗ Error loading documents: {e}")
        raise

    # Create vector store and index
    try:
        # Use simple vector store instead of Milvus
        vector_store = SimpleVectorStore()
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        vector_index = VectorStoreIndex.from_documents(
            docs,
            show_progress=True,
            transformations=[node_parser],
            storage_context=storage_context,
        )
        print("✓ Vector index created")
    except Exception as e:
        print(f"✗ Error creating vector index: {e}")
        raise

    # Custom prompt template for better responses
    template = (
        "You are a knowledgeable assistant analyzing documents. Your task is to answer "
        "the user's question based on the provided context from the documents.\n\n"
        "Guidelines:\n"
        "1. Base your answer primarily on the provided context\n"
        "2. If the context contains relevant information, provide a comprehensive answer\n"
        "3. If the context doesn't contain sufficient information, state this clearly\n"
        "4. Synthesize information from multiple parts of the context when relevant\n"
        "5. Be precise and factual in your response\n\n"
        "Context: {context_str}\n\n"
        "Question: {query_str}\n\n"
        "Answer: "
    )
    qa_template = PromptTemplate(template)

    # Create query engine
    docs_query_engine = vector_index.as_query_engine(
        text_qa_template=qa_template, 
        similarity_top_k=5
    )

    # Define the document query function
    def document_query_tool(query: str):
        """Query documents using semantic search."""
        try:
            print(f"🔍 Searching documents for: {query[:50]}...")
            response_obj = docs_query_engine.query(query)
            response = str(response_obj)
            print("✓ Document search completed")
            return response
        except Exception as e:
            error_msg = f"Error querying documents: {str(e)}"
            print(f"✗ {error_msg}")
            return error_msg

    # Create tool for document querying
    docs_tool = FunctionTool.from_defaults(
        document_query_tool,
        name="document_tool",
        description=(
            "Use this tool to search and analyze uploaded documents (PDFs, Word docs, etc.). "
            "This tool is for questions about:\n"
            "- Policies, procedures, and guidelines\n"
            "- Terms and conditions, privacy policies\n"
            "- College/institutional information\n"
            "- Conceptual explanations and qualitative content\n"
            "- Any text-based content from uploaded documents\n\n"
            "Do NOT use this tool for:\n"
            "- Finding specific people's personal data (use sql_tool instead)\n"
            "- Structured data queries about names, birthdays, phone numbers, etc.\n\n"
            "Examples of queries for this tool:\n"
            "- 'What is the privacy policy?'\n"
            "- 'Tell me about the college principles'\n"
            "- 'What are the terms and conditions?'"
        ),
    )

    print("✓ Document tool created")
    return docs_tool


def get_database_schema(db_path, table_name):
    """Get schema information for the database table."""
    try:
        engine = create_engine(f"sqlite:///{db_path}")
        
        # Get column information
        from sqlalchemy import text
        with engine.connect() as conn:
            result = conn.execute(text(f"PRAGMA table_info({table_name})"))
            schema_info = result.fetchall()
            
            columns = []
            for row in schema_info:
                columns.append({
                    'name': row[1],
                    'type': row[2],
                    'not_null': bool(row[3]),
                    'primary_key': bool(row[5])
                })
            
            return columns
    except Exception as e:
        print(f"✗ Error getting database schema: {e}")
        return []


def validate_tools(sql_tool, doc_tool):
    """Validate that tools are properly configured."""
    try:
        # Test SQL tool
        print("🧪 Testing SQL tool...")
        # This is a basic validation - in production you might want more thorough testing
        assert sql_tool.metadata.name == "sql_tool"
        print("✓ SQL tool validation passed")
        
        # Test document tool
        print("🧪 Testing document tool...")
        assert doc_tool.metadata.name == "document_tool"
        print("✓ Document tool validation passed")
        
        return True
    except Exception as e:
        print(f"✗ Tool validation failed: {e}")
        return False