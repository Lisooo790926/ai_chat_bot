"""
This module contains functions for managing vector storage data in Qdrant.
"""

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, HnswConfigDiff, OptimizersConfigDiff
from dotenv import dotenv_values
from utils.logger import logger

class VectorStoreManager:
    def __init__(self):
        self.config = dotenv_values(".env")
        self.client = QdrantClient(url=self.config.get("QDRANT_URL", "http://localhost:6333"))

    def clean_collection_by_dataset(self, collection_name: str, dataset_name: str) -> tuple[int, str]:
        """
        Clean specific dataset data from a collection.
        
        Args:
            collection_name: Name of the collection
            dataset_name: Name of the dataset to clean
            
        Returns:
            tuple: (number of deleted points, status message)
        """
        try:
            collections = self.client.get_collections().collections
            collection_exists = any(c.name == collection_name for c in collections)
            
            if not collection_exists:
                logger.error(f"Collection '{collection_name}' does not exist")
                return 0, f"Collection '{collection_name}' does not exist"

            result = self.client.delete(
                collection_name=collection_name,
                points_selector=self.client.points_selector(
                    filter={
                        "must": [
                            {
                                "key": "metadata.dataset",
                                "match": {"value": dataset_name}
                            }
                        ]
                    }
                )
            )
            
            logger.info(f"Successfully cleaned {result.status.deleted_count} entries from {collection_name}")
            return result.status.deleted_count, "Success"

        except Exception as e:
            logger.error(f"Error cleaning collection: {str(e)}")
            return 0, f"Error: {str(e)}"

    def list_collections(self) -> list[str]:
        """Get list of all collections"""
        try:
            collections = self.client.get_collections().collections
            return [c.name for c in collections]
        except Exception as e:
            logger.error(f"Error listing collections: {str(e)}")
            return []

    def get_collection_info(self, collection_name: str) -> dict:
        """Get information about a specific collection"""
        try:
            return self.client.get_collection(collection_name=collection_name).model_dump()
        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}")
            return {}

    def create_collection(
        self, 
        collection_name: str, 
        vector_size: int, 
        force_recreate: bool = False,
        m: int = 16,
        ef_construct: int = 100,
        indexing_threshold: int = 20000,
        on_disk: bool = True
    ) -> bool:
        """
        Create a new collection with HNSW index configuration
        
        Args:
            collection_name: Name of the collection
            vector_size: Size of vectors
            force_recreate: Whether to recreate if exists
            m: Number of edges per node (default: 16, higher = better recall but more memory)
            ef_construct: Build time accuracy vs speed (default: 100, higher = more accurate but slower to build)
            indexing_threshold: Minimal number of vectors for automatic index building
            on_disk: Whether to store payload on disk
        """
        try:
            if force_recreate:
                try:
                    self.client.delete_collection(collection_name)
                except Exception:
                    pass

            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                    # HNSW index configuration
                    hnsw_config=HnswConfigDiff(
                        m=m,  # Number of edges per node
                        ef_construct=ef_construct,  # Build time accuracy
                    )
                ),
                optimizers_config=OptimizersConfigDiff(
                    indexing_threshold=indexing_threshold,  # When to build index
                ),
                on_disk_payload=on_disk
            )
            return True
        except Exception as e:
            logger.error(f"Error creating collection: {str(e)}")
            return False

    def get_collection_name(self, base_name: str, provider: str = "gemini") -> str:
        """Get provider-specific collection name"""
        return f"{base_name}_{provider}" if provider != "azure" else base_name

    def collection_exists(self, collection_name: str) -> bool:
        """Check if collection exists"""
        try:
            collections = self.client.get_collections().collections
            return any(c.name == collection_name for c in collections)
        except Exception as e:
            logger.error(f"Error checking collection existence: {str(e)}")
            return False 