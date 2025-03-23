"""
This module contains the code to embed a PDF file into a Qdrant collection.
"""

import glob
from langchain.docstore.document import Document
from langchain_openai import AzureOpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import dotenv_values
from .vector_store import VectorStoreManager
from utils.logger import logger

def embed_text(collection, dataset, markdown_file, provider="gemini", force_recreate=False):
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
    vector_manager = VectorStoreManager()

    # Get provider-specific collection name
    collection_name = vector_manager.get_collection_name(collection, provider)

    # Get the appropriate embedding model
    embedding_llm, vector_size = get_embedding_model(provider)

    # Create or recreate collection if needed
    if force_recreate or not vector_manager.collection_exists(collection_name):
        vector_manager.create_collection(collection_name, vector_size, force_recreate)

    doc = Document(
        page_content=markdown_text, metadata={"dataset": dataset, "file": markdown_file}
    )

    markdown_splitter = RecursiveCharacterTextSplitter(
        separators=["#", "##", "###", "\n\n", "\n", " "],
        chunk_size=200,  # 可根據需求調整 chunk 大小
        chunk_overlap=20,  # 重疊區域，避免語境斷裂
    )
    documents = markdown_splitter.split_documents([doc])

    try:
        logger.info(f"Starting embedding process for {markdown_file}")
        # return a vector store
        qdrant = QdrantVectorStore.from_documents(
            documents,
            embedding=embedding_llm,
            url=config.get("QDRANT_URL", "http://localhost:6333"),
            collection_name=collection_name,
        )

        if force_recreate:
            logger.info(f"Recreating collection {collection_name}")
            # ... recreation code ...
            
        logger.info(f"Successfully embedded content into {collection_name}")
        return qdrant
    except Exception as e:
        logger.error(f"Error embedding documents: {str(e)}")
        if "429" in str(e):
            logger.critical("API quota exceeded - immediate attention required")
        if "dimensions" in str(e) and not force_recreate:
            logger.info("Dimensions mismatch detected. Recreating collection...")
            return embed_text(collection, dataset, markdown_file, provider, force_recreate=True)
        raise

def get_embedding_model(provider="azure"):

    """
    Get embedding model based on provider choice
    Returns:
        tuple: (embedding_model, vector_size)
    """
    config = dotenv_values(".env")
    
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
            model="models/gemini-embedding-exp-03-07",
            google_api_key=config.get("GOOGLE_API_KEY"),
        )
        vector_size = 3072  # Google embedding size
    
    return embedding_llm, vector_size
