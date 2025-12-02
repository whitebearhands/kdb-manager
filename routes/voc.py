from fastapi import APIRouter, Body
from config import settings
from qdrant_client import AsyncQdrantClient
from qdrant_client import models

from modules.dependencies import get_embedding_model

router = APIRouter(prefix="/api", tags=["VOC CACHE"])

qdrant_client = AsyncQdrantClient(
    url=settings.qdrant.url, api_key=settings.qdrant.api_key
)
embedding_model = get_embedding_model()


@router.post("/v1/voc/get")
async def get_similar_queries(collection_name: str = Body(...), query: str = Body(...)):
    dense_query = embedding_model.encode(query, normalize_embeddings=True).tolist()
    results = await qdrant_client.search(
        collection_name=collection_name,
        query_vector=models.NamedVector(name="dense", vector=dense_query),
        limit=3,
        score_threshold=0.8,
    )

    return [{"score": r.score, "payload": r.payload} for r in results]
