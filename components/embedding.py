"""
This module contains the code to embed a PDF file into a Qdrant collection.
"""

import glob
from langchain.docstore.document import Document
from langchain_openai import AzureOpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from dotenv import dotenv_values


def embed_text(collection, dataset, markdown_file, provider="azure", force_recreate=False):
    """
    Embeds the text from a file into a Qdrant collection.
    Args:
        collection: The name of the Qdrant collection to create.
        dataset: The name of the dataset this document belongs to.
        markdown_file: The path to the markdown file containing the text to embed.
        provider: 'azure' or 'gemini' to specify which embedding provider to use.
        force_recreate: Whether to force recreate the collection with dimensions matching the embedding model.
    """
    with open(markdown_file, "r", encoding="utf-8") as f:
        markdown_text = f.read()

    config = dotenv_values(".env")
    client = QdrantClient(url="http://localhost:6333")

    collection_name = f"{collection}_{provider}" if provider != "azure" else collection

    # Get the appropriate embedding model
    if provider == "azure":
        embedding_llm = AzureOpenAIEmbeddings(
            azure_endpoint=config.get("AZURE_OPENAI_ENDPOINT"),
            azure_deployment=config.get("AZURE_OPENAI_Embedding_DEPLOYMENT_NAME"),
            api_key=config.get("AZURE_OPENAI_KEY"),
            openai_api_version=config.get("AZURE_OPENAI_API_VERSION"),
        )
        vector_size = 1536  # OpenAI embedding size
    else:
        embedding_llm = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=config.get("GOOGLE_API_KEY"),
        )
        vector_size = 768  # Google embedding size

    # Force recreate if requested or if dimensions don't match
    if force_recreate:
        try:
            client.delete_collection(collection_name)
        except Exception:
            # Collection might not exist yet
            pass
        
        # Create collection with correct dimensions
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "size": vector_size,
                "distance": "Cosine"
            },
            optimizers_config={
                "indexing_threshold": 20000
            },
            on_disk_payload=True
        )

    doc = Document(
        page_content=markdown_text, metadata={"dataset": dataset, "file": markdown_file}
    )

    markdown_splitter = RecursiveCharacterTextSplitter(
        separators=["#", "##", "###", "\n\n", "\n", " "],
        chunk_size=1000,  # 可根據需求調整 chunk 大小
        chunk_overlap=100,  # 重疊區域，避免語境斷裂
    )
    documents = markdown_splitter.split_documents([doc])

    try:
        qdrant = QdrantVectorStore.from_documents(
            documents,
            embedding=embedding_llm,
            url="http://localhost:6333",
            collection_name=collection_name,
        )
        return qdrant
    except Exception as e:
        print(f"Error embedding documents: {e}")
        # If dimensions don't match, try forcing recreation
        if "dimensions" in str(e) and not force_recreate:
            print("Dimensions mismatch detected. Recreating collection...")
            return embed_text(collection, dataset, markdown_file, provider, force_recreate=True)
        raise
