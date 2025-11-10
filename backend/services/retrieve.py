import logging
from urllib import request
from pydantic import BaseModel
import cohere
from pinecone import Pinecone
import os
from dotenv import load_dotenv
load_dotenv(override=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from backend.utils.ai_workflow_utils.get_config_value import get_config_value
from backend.config.chatbot_config import EMBEDDING_CONFIG

# Cohere
cohere_api_key = os.getenv("COHERE_API_KEY")
co = cohere.ClientV2(cohere_api_key)
embedding_model_name = get_config_value(config_set=EMBEDDING_CONFIG, key="model")
# Pinecone
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)

class RetrieveRequest(BaseModel):
    query: str
    index_name: str
    namespace: str | None = None  # Required when "private" is selected
    libraries: list[str]  # e.g. ["organization", "private", "public"]
    top_k: int = 5

class Retriever:
    def __init__(self):
        pass

    def query_index(self, index_name: str, namespace: str | None, metadata_filter: dict | None, query_embedding, top_k):
        if not pc.has_index(index_name):
            print(f"[retrieve] Index not found: {index_name}")
            return []
        index = pc.Index(name=index_name)
        print(f"[retrieve] Querying Pinecone index: {index_name} namespace: {namespace}")
        response = index.query(
            namespace=namespace,
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            include_values=False,
            filter=metadata_filter
        )
        return response.get("matches", [])

    def retrieve(self, retrieve_request: RetrieveRequest):
        """Retrieve similar documents from the Pinecone vector store."""
        try:
            print("[retrieve] Start retrieval process")

            requested_libraries = set(retrieve_request.libraries)
            if not requested_libraries:
                raise ValueError("At least one library must be selected")

            allowed_libraries = {"organization", "private", "public"}
            invalid = requested_libraries - allowed_libraries
            if invalid:
                raise ValueError(f"Invalid library selections: {sorted(invalid)}")

            if "private" in requested_libraries and not retrieve_request.namespace:
                raise ValueError("user_id is required when querying the private library")

            print(f"[retrieve] Libraries requested: {sorted(requested_libraries)}")

            # Embed the query once
            print(f"[retrieve] Embedding query: {retrieve_request.query}")
            query_embedding = co.embed(texts=[retrieve_request.query],
                                    model=embedding_model_name,
                                    input_type="search_query",
                                    embedding_types=["float"]).embeddings.float_[0]

            aggregated_matches = []

            # Organization library
            if "organization" in requested_libraries:
                matches = self.query_index(
                    index_name=retrieve_request.index_name,
                    namespace="organization",
                    metadata_filter={"library": {"$eq": "organization"}},
                    query_embedding=query_embedding,
                    top_k=retrieve_request.top_k
                )
                aggregated_matches.extend(matches)

            # Private library
            if "private" in requested_libraries:
                matches = self.query_index(
                    index_name=retrieve_request.index_name,
                    namespace=retrieve_request.namespace,
                    metadata_filter={
                        "library": {"$eq": "private"},
                        "user_id": {"$eq": retrieve_request.namespace},
                    },
                    query_embedding=query_embedding,
                    top_k=retrieve_request.top_k
                )
                aggregated_matches.extend(matches)

            # public library (separate public index, namespace optional)
            if "public" in requested_libraries:
                matches = self.query_index(
                    index_name="public",
                    namespace=None,
                    metadata_filter={"library": {"$eq": "public"}},
                    query_embedding=query_embedding,
                    top_k=retrieve_request.top_k
                )
                aggregated_matches.extend(matches)

            if not aggregated_matches:
                print("[retrieve] No matches found across selected libraries")
                return {
                    "status": "success",
                    "query": retrieve_request.query,
                    "results": [],
                    "total_results": 0,
                }

            # Sort combined matches by descending score and trim to top_k / number of libraries
            aggregated_matches.sort(key=lambda match: match.get("score", 0), reverse=True)
            print(f"[retrieve] top k aggregated: {retrieve_request.top_k * len(requested_libraries)}")
            aggregated_matches = aggregated_matches[: retrieve_request.top_k * len(requested_libraries)] # //
            # ! Temporary change: return 5 top_k, not divided by libraries because it's only from public
            # aggregated_matches = aggregated_matches[:5]

            retrieved_docs = []
            for match in aggregated_matches:
                metadata = match.get("metadata", {})
                print(f"[retrieve] Match metadata: {metadata}")
                retrieved_docs.append({
                    "chunk_id": match.get("chunk_id"),
                    "chunk_text": metadata.get("chunk_text", ""),
                    "score": match.get("score"),
                    "page": metadata.get("page"),
                    "library": metadata.get("library"),
                    "doc_name": metadata.get("doc_name", ""),
                    "storage_path": metadata.get("storage_path", "")
                })

            print(f"[retrieve] Returning {len(retrieved_docs)} results")
            return {
                "status": "success",
                "query": retrieve_request.query,
                "results": retrieved_docs,
                "total_results": len(retrieved_docs),
            }
        except Exception as e:
            logger.error(f"Error during retrieval: {str(e)}")
            raise