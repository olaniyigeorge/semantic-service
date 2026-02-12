from __future__ import annotations

from typing import Any, List, Optional
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from qdrant_client import QdrantClient
from semantic_core.models import EmbeddedChunk, SearchQuery, SearchResult
from semantic_core.vectorstores.base import VectorStore
import uuid


class QDrantVectorStore(VectorStore):
    """
    Qdrant-backed vector store implementation.
    """

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        embedding_model: Any,  
        vector_size: Optional[int] = None,
    ) -> None:
        """
        Initialize the Qdrant vector store.
        
        Args:
            client: QdrantClient instance
            collection_name: Name of the collection to use
            embedding_model: Embedding model for generating vectors
            vector_size: Size of the embedding vectors (auto-detected if None)
        """
        self.client = client
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        
        # Auto-detect vector size if not provided
        if vector_size is None:
            print("Auto-detecting vector size...")
            # Use a longer sample text to get actual embedding size
            sample_vector = embedding_model.embed_texts(
                "This is a longer sample text to ensure we get the correct embedding dimensions."
            )
            vector_size = len(self._extract_vector(sample_vector))
            print(f"Detected vector size: {vector_size}")
        
        self.vector_size = vector_size
        
        # Create collection if it doesn't exist
        if not client.collection_exists(collection_name):
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size if vector_size < 128 else 768, 
                    distance=Distance.COSINE
                )
            )

    def _extract_vector(self, vector) -> List[float]:
        """
        Extract vector values from various formats.
        
        Handles:
        - ContentEmbedding objects (Vertex AI)
        - Plain lists
        - Numpy arrays
        - Other embedding objects with .values attribute
        
        Args:
            vector: Vector in any supported format
            
        Returns:
            List of float values
            
        Raises:
            ValueError: If vector format is not supported
        """
        # Case 1: ContentEmbedding or similar object with .values attribute
        if hasattr(vector, 'values'):
            values = vector.values
            # Ensure it's a list of floats
            if isinstance(values, list):
                return [float(v) for v in values]
            # Handle numpy array
            elif hasattr(values, 'tolist'):
                return [float(v) for v in values.tolist()]
            else:
                return [float(v) for v in list(values)]
        
        # Case 2: Already a list
        elif isinstance(vector, list):
            # Handle list of ContentEmbedding objects or similar with .values
            if vector and hasattr(vector[0], 'values'):
                return self._extract_vector(vector[0].values)
            # Handle list of floats/numbers
            else:
                return [float(v) for v in vector]
        
        # Case 3: Numpy array
        elif hasattr(vector, 'tolist'):
            return [float(v) for v in vector.tolist()]
        
        # Case 4: Tuple (convert to list)
        elif isinstance(vector, tuple):
            return [float(v) for v in vector]
        
        # Case 5: Unknown format
        else:
            raise ValueError(
                f"Unsupported vector format: {type(vector)}. "
                f"Expected list, numpy array, or object with .values attribute. "
                f"Got: {vector}"
            )

    def upsert(self, items: List[EmbeddedChunk]) -> None:
        """
        Insert or update embedded chunks in the underlying store.
        
        Args:
            items: List of EmbeddedChunk objects to upsert
        """
        if not items:
            print.warning("No items to upsert")
            return

        print(f"Upserting {len(items)} items to Qdrant collection '{self.collection_name}'")
        
        points = []
        for item in items:
            # Get the chunk_id (SHA256 hash string)
            chunk_id = getattr(item, 'chunk_id', None) or getattr(item.chunk, 'chunk_id', None)
            
            # Generate a deterministic UUID from the chunk_id
            if chunk_id:
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))
            else:
                point_id = str(uuid.uuid4())
            
            # Extract the actual vector values
            # Handle different vector formats (ContentEmbedding, list, numpy array, etc.)
            vector = self._extract_vector(item.vector)

            # Create payload with metadata
            payload = {
                "doc_id": item.chunk.doc_id,
                "chunk_id": point_id,
                "text": item.chunk.text,
                "metadata": getattr(item.chunk, 'metadata', {}),
                "page_number": getattr(item.chunk, 'page_number', None),
                "start_char": getattr(item.chunk, 'start_char', None),
                "end_char": getattr(item.chunk, 'end_char', None),
            }
            
            # Create point with vector and payload
            point = PointStruct(
                id=point_id,
                vector=vector, 
                payload=payload
            )
            
            points.append(point)
        
        try:
            # Upsert points to collection
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            print(f"Successfully upserted {len(points)} points to collection '{self.collection_name}'")
        except Exception as e:
            print(f"Error upserting to Qdrant: {e}")
            raise



    def delete_by_doc(self, doc_id: str) -> None:
        """
        Remove all chunks belonging to the given logical document.
        
        Args:
            doc_id: Document ID to delete all chunks for
        """
        # Delete points with matching doc_id in payload
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="doc_id",
                        match=MatchValue(value=doc_id)
                    )
                ]
            )
        )

    def query(
        self,
        qvec: List[float],
        query: SearchQuery
    ) -> List[SearchResult]:
        """
        Run a vector similarity search using the provided query vector and
        additional filtering / ranking parameters.
        
        Args:
            qvec: Query vector for similarity search
            query: SearchQuery object with additional parameters
            
        Returns:
            List of SearchResult objects
        """
        # Build filter if needed
        query_filter = None
        if hasattr(query, 'filters') and query.filters:
            conditions = []
            for key, value in query.filters.items():
                conditions.append(
                    FieldCondition(
                        key=f"metadata.{key}",
                        match=MatchValue(value=value)
                    )
                )
            if conditions:
                query_filter = Filter(must=conditions)
        
        # Determine limit
        limit = getattr(query, 'top_k', 10)
        
        # Perform search
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=qvec,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False  # Set to True if you need vectors in results
        )
        
        # Convert to SearchResult objects
        results = []
        for hit in search_results:
            result = SearchResult(
                doc_id=hit.payload.get("doc_id"),
                chunk_id=hit.payload.get("chunk_id"),
                text=hit.payload.get("text"),
                score=hit.score,
                metadata=hit.payload.get("metadata", {}),
                vector=hit.vector if hasattr(hit, 'vector') else None
            )
            results.append(result)
        
        return results

    def delete_collection(self) -> None:
        """
        Delete the entire collection (useful for cleanup).
        """
        self.client.delete_collection(collection_name=self.collection_name)

    def get_collection_info(self) -> dict:
        """
        Get information about the collection.
        
        Returns:
            Dictionary with collection information
        """
        return self.client.get_collection(collection_name=self.collection_name)