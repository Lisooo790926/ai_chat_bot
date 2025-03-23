from typing import List, Optional, AsyncGenerator
import traceback
import logging
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain import schema
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from langchain_qdrant import QdrantVectorStore
from dotenv import dotenv_values
from pydantic import BaseModel
from .embedding import get_embedding_model, get_vector_store_client

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Core models that can be used by both API and gRPC
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    query: str
    chat_history: Optional[List[ChatMessage]] = []
    dataset_name: str
    provider: str = "azure"

class ChatResponse(BaseModel):
    response: str

class RAGBot:
    def __init__(self):
        self.config = dotenv_values(".env")
        self.datasets = self.config.get("QDRANT_DATASETS").split(",")
        self.system_prompt = (
            "You are an AI assistant that answers questions based on provided documents. If you don't know the answer based on the documents, say you don't know."
            "Please answer the question based on the following reference materials:"
            "Chat history: {chat_history}"
            "References: {context}"
        )
        self.client = get_vector_store_client()
    
    def get_llm_and_embeddings(self, provider="azure"):
        """Get LLM and embeddings models based on provider choice"""
        if provider == "azure":
            generator_llm = AzureChatOpenAI(
                azure_endpoint=self.config.get("AZURE_OPENAI_ENDPOINT"),
                azure_deployment=self.config.get("AZURE_OPENAI_DEPLOYMENT_NAME"),
                openai_api_version=self.config.get("AZURE_OPENAI_API_VERSION"),
                api_key=self.config.get("AZURE_OPENAI_KEY"),
                streaming=True,
            )
        else:
            generator_llm = GoogleGenerativeAI(
                model="gemini-2.0-flash",
                google_api_key=self.config.get("GOOGLE_API_KEY"),
                temperature=0.7,
                streaming=True,
            )
        
        embedding_llm, _ = get_embedding_model(provider)
        return generator_llm, embedding_llm

    async def get_response_stream(
        self,
        user_query: str,
        chat_history: List[schema.HumanMessage],
        collection_name: str,
        dataset_name: str,
        provider: str,
    ) -> AsyncGenerator[str, None]:
        """Async generator for streaming responses"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{input}"),
        ])
        
        try:
            generator_llm, embedding_llm = self.get_llm_and_embeddings(provider)
           
            # Determine provider-specific collection name
            provider_collection = f"{collection_name}_{provider}" if provider != "azure" else collection_name
            logger.info(f"Using collection: {provider_collection} with provider: {provider}")
            
            try:
                # Try to use the provider-specific collection first
                qdrant = QdrantVectorStore(
                    client=self.client, 
                    collection_name=provider_collection, 
                    embedding=embedding_llm
                )
            except Exception as e:
                qdrant = self.create_collection_if_not_found(provider, provider_collection, collection_name, e)

            if qdrant is None:
                raise Exception("Failed to create collection")
            
            question_answer_chain = create_stuff_documents_chain(generator_llm, prompt)
            response_stream = self.generate_response(qdrant, user_query, chat_history, dataset_name, question_answer_chain)
            
            for chunk in response_stream:
                yield chunk

        except Exception as e:
            error_detail = traceback.format_exc()
            logger.error(f"Unexpected error in response stream: {str(e)}\n{error_detail}")
            yield f"An unexpected error occurred: {str(e)}\n\nDebug info: {type(e).__name__}"

    async def generate_response(self, qdrant, user_query: str, chat_history: List[schema.HumanMessage], dataset_name: str, question_answer_chain):

        retriever = qdrant.as_retriever(
            search_kwargs=dict(
                k=5, # number of chunks to retrieve
                filter=Filter(
                    min_should=2,
                    should=[
                        FieldCondition(
                            key="metadata.dataset",
                            match=MatchValue(value=dataset_name),
                        )
                    ]
                ),
            )
        )

        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        chain = rag_chain.pick("answer")
        
        try:
            response_stream = chain.stream({"input": user_query, "chat_history": chat_history})
            return response_stream
        except Exception as e:
            error_detail = traceback.format_exc()
            logger.error(f"Error generating response: {str(e)}\n{error_detail}")
            return f"Error generating response: {str(e)}\n\nDebug info: {type(e).__name__}"

    def create_collection_if_not_found(self, provider: str, provider_collection: str, default_collection: str, e: Exception):
        # If provider-specific collection doesn't exist, create it
        error_detail = traceback.format_exc()
        logger.warning(f"Error accessing collection: {str(e)}\n{error_detail}")
        
        try:
            if "Not found: Collection" in str(e):
                logger.info(f"Collection {provider_collection} doesn't exist. Creating it.")
                # Get vector size based on provider
                embedding_llm, vector_size = get_embedding_model(provider)
                
                # Create the collection with the right dimensions
                self.client.create_collection(
                    collection_name=provider_collection,
                    vectors_config={
                        "size": vector_size,
                        "distance": "Cosine"
                    },
                    optimizers_config={
                        "indexing_threshold": 20000
                    },
                    on_disk_payload=True
                )
                
                # Create empty vector store
                qdrant = QdrantVectorStore(
                    client=self.client, 
                    collection_name=provider_collection, 
                    embedding=embedding_llm
                )
                return qdrant
            else:
                # For other errors, fall back to the main collection
                logger.warning(f"Falling back to main collection: {provider_collection}")
                qdrant = QdrantVectorStore(
                    client=self.client, 
                    collection_name=default_collection, 
                    embedding=embedding_llm
                )
                return qdrant

        except Exception as inner_e:
            # If all else fails, return a helpful error message
            inner_error_detail = traceback.format_exc()
            logger.error(f"Error setting up vector store: {str(inner_e)}\n{inner_error_detail}")
            yield f"I'm having trouble accessing the knowledge base. Error: {str(inner_e)}\n\nDebug info: {type(inner_e).__name__}"
            return
    
    def validate_dataset(self, dataset_name: str) -> bool:
        """Validate if dataset exists"""
        return dataset_name in self.datasets

    def convert_chat_history(self, chat_history: List[ChatMessage]) -> List[schema.HumanMessage]:
        """Convert API chat history format to LangChain format"""
        history = []
        for msg in chat_history:
            if msg.role == "ai":
                history.append(AIMessage(content=msg.content))
            else:
                history.append(HumanMessage(content=msg.content))
        return history 