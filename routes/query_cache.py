"""
routes/query_cache.py
=====================
쿼리 캐싱 엔드포인트.

담당 기능
---------
- 쿼리 추가  : 쿼리를 dense 벡터로 임베딩하여 query_cache 컬렉션에 저장
- 유사 쿼리  : 입력 쿼리와 유사한 기존 쿼리 목록 반환 (자동완성/추천 용도)

구조 메모
---------
  query_cache 컬렉션은 자동 생성되며, dense 벡터만 사용한다.
  컬렉션 단위로 쿼리를 분리 저장하므로, 다수의 RAG 컬렉션을 동시에 운용해도 혼용되지 않는다.
"""

import asyncio
import uuid
from datetime import datetime
from logging import getLogger

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from qdrant_client.http import models

from modules.dependencies import get_embedding_model, get_qdrant_client

logger = getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Query Cache"])

# ── 모듈 레벨 싱글톤 ────────────────────────────────────────────────────────
qdrant_client = get_qdrant_client()
embedding_model = get_embedding_model()

QUERY_CACHE_COLLECTION = "query_cache"


# ════════════════════════════════════════════════════════════════════════════════
# 내부 헬퍼
# ════════════════════════════════════════════════════════════════════════════════


async def _ensure_query_cache_collection() -> None:
    """
    query_cache 컬렉션이 없으면 생성한다.

    벡터 설정
    ---------
    - dense : COSINE 거리, HNSW (m=16, ef_construct=200)
    - payload 인덱스: collection (KEYWORD) — 컬렉션별 필터 성능 보장
    """
    is_exist = await qdrant_client.collection_exists(QUERY_CACHE_COLLECTION)
    if is_exist:
        return

    dimension = embedding_model.get_sentence_embedding_dimension()
    await qdrant_client.create_collection(
        collection_name=QUERY_CACHE_COLLECTION,
        vectors_config={
            "dense": models.VectorParams(
                size=dimension,
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
        collection_name=QUERY_CACHE_COLLECTION,
        field_name="collection",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
    logger.info(f"query_cache 컬렉션 생성 완료 (dim={dimension})")


# ════════════════════════════════════════════════════════════════════════════════
# 엔드포인트
# ════════════════════════════════════════════════════════════════════════════════


@router.post(
    "/v1/query/add",
    operation_id="add_query_cache",
    summary="쿼리 캐시 추가",
)
async def add_query(collection_name: str, query: str):
    """
    쿼리를 dense 벡터로 임베딩하여 query_cache 컬렉션에 저장한다.

    포인트 payload 구조
    -------------------
    {
        "query": "<쿼리 텍스트>",
        "collection": "<RAG 컬렉션 이름>",
        "date_time": "<ISO 8601 타임스탬프>"
    }
    """
    await _ensure_query_cache_collection()

    # CPU 바운드 임베딩을 별도 스레드에서 실행
    dense_embedding = await asyncio.to_thread(
        embedding_model.encode, query, normalize_embeddings=True
    )

    result = await qdrant_client.upsert(
        collection_name=QUERY_CACHE_COLLECTION,
        points=[
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector={"dense": dense_embedding.tolist()},
                payload={
                    "query": query,
                    "collection": collection_name,
                    "date_time": datetime.now().isoformat(),
                },
            )
        ],
    )
    logger.info(f"쿼리 캐시 추가: collection={collection_name}, query={query[:50]}")
    return JSONResponse(content=result.model_dump())


@router.post(
    "/v1/query/get",
    operation_id="get_similar_queries",
    summary="유사 쿼리 검색",
)
async def get_similar_queries(collection_name: str, query: str):
    """
    입력 쿼리와 의미적으로 유사한 캐시된 쿼리 목록을 반환한다.

    - 상위 10개 반환, score_threshold=0.1 (너무 다른 쿼리 제외)
    - 자동완성, 검색어 추천 등 UX 용도로 활용

    Returns
    -------
    List[str]
        유사 쿼리 텍스트 목록 (점수 순)
    """
    is_exist = await qdrant_client.collection_exists(QUERY_CACHE_COLLECTION)
    if not is_exist:
        return []

    # CPU 바운드 임베딩을 별도 스레드에서 실행
    dense_query = await asyncio.to_thread(
        embedding_model.encode, query, normalize_embeddings=True
    )

    results = await qdrant_client.query_points(
        collection_name=QUERY_CACHE_COLLECTION,
        query=dense_query.tolist(),
        using="dense",
        limit=10,
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="collection",
                    match=models.MatchValue(value=collection_name),
                )
            ]
        ),
        search_params=models.SearchParams(hnsw_ef=128),
        score_threshold=0.1,
    )

    return [point.payload["query"] for point in results.points]
