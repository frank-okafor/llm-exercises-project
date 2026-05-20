import chromadb
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_embedding_client():
    """Get OpenAI client for embeddings"""
    return OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://openai.vocareum.com/v1")
    )

def get_query_embedding(query: str) -> List[float]:
    """Generate embedding for a query string"""
    client = get_embedding_client()
    model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    response = client.embeddings.create(
        input=query,
        model=model
    )
    return response.data[0].embedding

def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    """Discover available ChromaDB backends in the project directory"""
    backends = {}
    current_dir = Path(".")
    
    # Create list of directories that match specific criteria (directory type and name pattern)
    chroma_dirs = [d for d in current_dir.iterdir()
                   if d.is_dir() and ('chroma' in d.name.lower() or d.name.endswith('_db'))]

    # Also check for the default chroma_db directory
    default_dir = Path("./chroma_db")
    if default_dir.exists() and default_dir not in chroma_dirs:
        chroma_dirs.append(default_dir)

    # Loop through each discovered directory
    for chroma_dir in chroma_dirs:
        # Wrap connection attempt in try-except block for error handling
        try:
            # Initialize database client with directory path and configuration settings
            client = chromadb.PersistentClient(path=str(chroma_dir))

            # Retrieve list of available collections from the database
            collections = client.list_collections()

            # Loop through each collection found
            for collection in collections:
                # Create unique identifier key combining directory and collection names
                key = f"{chroma_dir.name}_{collection.name}"

                # Get document count with fallback for unsupported operations
                try:
                    doc_count = collection.count()
                except Exception:
                    doc_count = "Unknown"

                # Build information dictionary
                backends[key] = {
                    # Store directory path as string
                    "directory": str(chroma_dir),
                    # Store collection name
                    "collection_name": collection.name,
                    # Create user-friendly display name
                    "display_name": f"{collection.name} ({chroma_dir.name}) - {doc_count} docs",
                    # Store document count
                    "doc_count": doc_count
                }

        # Handle connection or access errors gracefully
        except Exception as e:
            # Create fallback entry for inaccessible directories
            key = f"{chroma_dir.name}_error"
            backends[key] = {
                "directory": str(chroma_dir),
                "collection_name": "",
                # Include error information in display name with truncation
                "display_name": f"{chroma_dir.name} (Error: {str(e)[:50]}...)",
                # Set appropriate fallback values for missing information
                "doc_count": 0
            }

    # Return complete backends dictionary with all discovered collections
    return backends

def initialize_rag_system(chroma_dir: str, collection_name: str) -> Tuple[Any, bool, str]:
    """Initialize the RAG system with specified backend (cached for performance)"""
    try:
        # Create a chromadb persistent client
        client = chromadb.PersistentClient(path=chroma_dir)

        # Return the collection with the collection_name
        collection = client.get_collection(name=collection_name)

        return collection, True, ""
    except Exception as e:
        return None, False, str(e)

def retrieve_documents(collection, query: str, n_results: int = 3, 
                      mission_filter: Optional[str] = None) -> Optional[Dict]:
    """Retrieve relevant documents from ChromaDB with optional filtering"""

    # Initialize filter variable to None (represents no filtering)
    where_filter = None

    # Check if filter parameter exists and is not set to "all" or equivalent
    if mission_filter and mission_filter.lower() not in ["all", "none", ""]:
        # Create filter dictionary with appropriate field-value pairs
        # Normalize mission name to match stored format
        normalized_mission = mission_filter.lower().replace(" ", "_")
        where_filter = {"mission": normalized_mission}

    # Get query embedding
    query_embedding = get_query_embedding(query)

    # Execute database query with the following parameters:
    results = collection.query(
        # Pass search query in the required format (as embedding)
        query_embeddings=[query_embedding],
        # Set maximum number of results to return
        n_results=n_results,
        # Apply conditional filter (None for no filtering, dictionary for specific filtering)
        where=where_filter
    )

    # Return query results to caller
    return results

def format_context(documents: List[str], metadatas: List[Dict]) -> str:
    """Format retrieved documents into context"""
    if not documents:
        return ""
    
    # Initialize list with header text for context section
    context_parts = ["=== Retrieved Context from NASA Documents ===\n"]

    # Loop through paired documents and their metadata using enumeration
    for i, (doc, metadata) in enumerate(zip(documents, metadatas)):
        # Extract mission information from metadata with fallback value
        mission = metadata.get('mission', 'Unknown Mission')
        # Clean up mission name formatting (replace underscores, capitalize)
        mission_display = mission.replace('_', ' ').title()

        # Extract source information from metadata with fallback value
        source = metadata.get('source', 'Unknown Source')

        # Extract category information from metadata with fallback value
        category = metadata.get('document_category', 'general')
        # Clean up category name formatting (replace underscores, capitalize)
        category_display = category.replace('_', ' ').title()

        # Create formatted source header with index number and extracted information
        source_header = f"\n[Source {i + 1}]\nMission: {mission_display}\nFile: {source}\nCategory: {category_display}\nContent:"

        # Add source header to context parts list
        context_parts.append(source_header)

        # Check document length and truncate if necessary
        max_doc_length = 2000
        if len(doc) > max_doc_length:
            truncated_doc = doc[:max_doc_length] + "...[truncated]"
            # Add truncated document content to context parts list
            context_parts.append(truncated_doc)
        else:
            # Add full document content to context parts list
            context_parts.append(doc)

    # Join all context parts with newlines and return formatted string
    return "\n".join(context_parts)

def build_source_list(metadatas: List[Dict]) -> List[str]:
    """Build a list of source citations from metadata"""
    sources = []
    for i, metadata in enumerate(metadatas):
        mission = metadata.get('mission', 'Unknown').replace('_', ' ').title()
        source = metadata.get('source', 'Unknown')
        category = metadata.get('document_category', 'general').replace('_', ' ').title()
        sources.append(f"[{i + 1}] {mission} - {source} ({category})")
    return sources
