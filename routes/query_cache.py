from datetime import date, datetime
import html
import math
import time
import uuid
from fastapi import APIRouter, Body, Depends, Query
from fastapi.responses import JSONResponse
from sentence_transformers import SentenceTransformer
from config import settings
from qdrant_client import AsyncQdrantClient
from qdrant_client import models

from modules.dependencies import get_chunk_id, get_embedding_model

router = APIRouter(prefix="/api", tags=["QUERY CACHE"])

qdrant_client = AsyncQdrantClient(
    url=settings.qdrant.url, api_key=settings.qdrant.api_key
)
embedding_model = get_embedding_model()


async def create_vector(collection_name="query_cache"):
    await qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": models.VectorParams(
                size=embedding_model.get_sentence_embedding_dimension(),
                distance=models.Distance.COSINE,
                hnsw_config=models.HnswConfigDiff(m=16, ef_construct=200),
            )
        },
        optimizers_config=models.OptimizersConfigDiff(
            indexing_threshold=50000,
            memmap_threshold=100000,
            default_segment_number=10,
            max_optimization_threads=8,
        ),
    )

    await qdrant_client.create_payload_index(
        collection_name=collection_name,
        field_name="collection",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )


async def create_collection():
    query_collection = f"query_cache"
    is_exist = await qdrant_client.collection_exists(collection_name=query_collection)
    if not is_exist:
        await create_vector(query_collection)

    return query_collection


@router.post("/v1/query/add")
async def add_query(collection_name: str, query: str):
    query_collection = await create_collection()

    dense_embedding = embedding_model.encode(query, normalize_embeddings=True).tolist()

    result = await qdrant_client.upsert(
        collection_name=query_collection,
        points=[
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector={"dense": dense_embedding},
                payload={
                    "query": query,
                    "collection": collection_name,
                    "date_time": datetime.now().isoformat(),
                },
            )
        ],
    )
    return JSONResponse(content=result.model_dump())


@router.post("/v1/query/get")
async def get_similar_queries(collection_name: str, query: str):
    query_collection = await create_collection()
    dense_query = embedding_model.encode(query, normalize_embeddings=True).tolist()

    results = await qdrant_client.search_points(
        collection_name=query_collection,
        query=dense_query,
        using="dense",
        limit=10,
        filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="collection", match=models.MatchValue(value=collection_name)
                )
            ]
        ),
        search_params=models.SearchParams(hnsw_ef=128),
        score_threshold=0.1,
    )

    return [r.payload["query"] for r in results.points]
