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

router = APIRouter()

client = AsyncQdrantClient(url=settings.qdrant.url, api_key=settings.qdrant.api_key)


async def create_vector(collection_name, embeddings: SentenceTransformer):
    await client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": models.VectorParams(
                size=embeddings.get_sentence_embedding_dimension(),
                distance=models.Distance.COSINE,
                hnsw_config=models.HnswConfigDiff(
                    m=64, ef_construct=1000, full_scan_threshold=50000
                ),
            )
        },
        optimizers_config=models.OptimizersConfigDiff(
            indexing_threshold=50000,
            memmap_threshold=100000,
            default_segment_number=10,
            max_optimization_threads=8,
        ),
    )

    await client.create_payload_index(
        collection_name=collection_name,
        field_name="metadatas.collection_name",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )


@router.get("/api/v1/feedback/{collection_name}/{search_type}")
async def get_all_feedback(
    collection_name: str,
    search_type: str,
    page: int = Query(1),
    page_size: int = Query(10),
):
    is_exist = await client.collection_exists(collection_name="feedback_cache")
    if not is_exist:
        return JSONResponse(status_code=404, content={"result": "no data"})

    decoded_collection_name = html.unescape(collection_name)
    filter = models.Filter(
        must=[
            models.FieldCondition(
                key="metadatas.collection_name",
                match=models.MatchValue(value=decoded_collection_name),
            ),
            models.FieldCondition(
                key="metadatas.search_type",
                match=models.MatchValue(value=search_type),
            ),
        ]
    )

    total_points = await client.count(
        collection_name="feedback_cache", count_filter=filter
    )
    total_pages = max(1, math.ceil(total_points.count / page_size))
    result = await client.query_points(
        collection_name="feedback_cache",
        query_filter=filter,
        limit=page_size,
        offset=page_size * (page - 1),
    )
    return {
        "page": [result.payload for result in result.points],
        "page_info": {
            "total_elements": total_points.count,
            "total_pages": total_pages,
            "page": page,
            "first": page == 1,
            "last": page == total_pages,
            "empty": len(result.points) == 0,
        },
    }


@router.post("/api/v1/feedback")
async def feedback(
    collection_name: str = Body(...),
    chat_id: str = Body(...),
    query: str = Body(...),
    answer: str = Body(...),
    embeddings: SentenceTransformer = Depends(get_embedding_model),
    get_chunk_id: int = Depends(get_chunk_id),
):
    is_exist = await client.collection_exists(collection_name="feedback_cache")
    if not is_exist:
        await create_vector("feedback_cache", embeddings)

    dense_embedding = embeddings.encode([query], normalize_embeddings=True)
    id = get_chunk_id()
    result = await client.upsert(
        collection_name="feedback_cache",
        points=[
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense": dense_embedding[0].tolist(),
                },
                payload={
                    "id": id,
                    "context": query,
                    "metadatas": {
                        "chat_id": chat_id,
                        "answer": answer,
                        "collection_name": collection_name,
                    },
                },
            )
        ],
    )

    return JSONResponse(content=result.model_dump())


@router.delete("/api/v1/feedback/{chat_id}")
async def delete_feedback(chat_id: str):
    is_exist = await client.collection_exists(collection_name="feedback_cache")
    if not is_exist:
        return JSONResponse(status_code=404, content={"result": "no data"})

    result = await client.delete(
        "feedback_cache",
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadatas.chat_id", match=models.MatchValue(value=chat_id)
                    )
                ]
            )
        ),
    )

    return JSONResponse(content=result.model_dump())
