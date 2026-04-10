"""
routes/feedback.py
==================
검색 피드백 저장 및 조회 엔드포인트.

담당 기능
---------
- 피드백 저장 : 쿼리+답변 쌍을 Qdrant feedback_cache 컬렉션에 임베딩하여 저장
- 피드백 조회 : 컬렉션/검색유형별 피드백 페이징 조회
- 피드백 삭제 : chat_id 기준 삭제

구조 메모
---------
  feedback_cache 컬렉션은 자동 생성되며, dense 벡터만 사용한다
  (하이브리드 검색 불필요 — 관리 목적 데이터).
"""

import asyncio
import html
import math
import uuid
from logging import getLogger
from typing import List, Dict

from fastapi import APIRouter, Body, Query
from fastapi.responses import JSONResponse
from qdrant_client.http import models

from modules.dependencies import get_embedding_model, get_qdrant_client

logger = getLogger(__name__)

router = APIRouter(tags=["Feedback"])

# ── 모듈 레벨 싱글톤 ────────────────────────────────────────────────────────
qdrant_client = get_qdrant_client()
embedding_model = get_embedding_model()

# feedback_cache 컬렉션 이름 (고정)
FEEDBACK_COLLECTION = "feedback_cache"


# ════════════════════════════════════════════════════════════════════════════════
# 내부 헬퍼
# ════════════════════════════════════════════════════════════════════════════════


async def _ensure_feedback_collection() -> None:
    """
    feedback_cache 컬렉션이 없으면 생성한다.

    벡터 설정
    ---------
    - dense : COSINE 거리, HNSW (m=16, ef_construct=200) — 소규모 컬렉션이므로 작게 설정
    - payload 인덱스: metadatas.collection_name, metadatas.search_type (KEYWORD)
    """
    is_exist = await qdrant_client.collection_exists(FEEDBACK_COLLECTION)
    if is_exist:
        return

    dimension = embedding_model.get_sentence_embedding_dimension()
    await qdrant_client.create_collection(
        collection_name=FEEDBACK_COLLECTION,
        vectors_config={
            "dense": models.VectorParams(
                size=dimension,
                distance=models.Distance.COSINE,
                hnsw_config=models.HnswConfigDiff(m=16, ef_construct=200),
            )
        },
    )
    # 필터 성능을 위한 payload 인덱스 생성
    await qdrant_client.create_payload_index(
        collection_name=FEEDBACK_COLLECTION,
        field_name="metadatas.collection_name",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
    await qdrant_client.create_payload_index(
        collection_name=FEEDBACK_COLLECTION,
        field_name="metadatas.search_type",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
    logger.info(f"feedback_cache 컬렉션 생성 완료 (dim={dimension})")


# ════════════════════════════════════════════════════════════════════════════════
# 엔드포인트
# ════════════════════════════════════════════════════════════════════════════════


@router.get(
    "/api/v1/feedback/{collection_name}/{search_type}",
    operation_id="get_all_feedback",
    summary="피드백 목록 페이징 조회",
)
async def get_all_feedback(
    collection_name: str,
    search_type: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=500),
):
    """
    특정 컬렉션+검색유형의 피드백을 페이징으로 반환한다.

    collection_name은 HTML 인코딩된 문자열로 전달할 수 있다.
    """
    is_exist = await qdrant_client.collection_exists(FEEDBACK_COLLECTION)
    if not is_exist:
        return JSONResponse(status_code=404, content={"result": "no data"})

    decoded = html.unescape(collection_name)

    query_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="metadatas.collection_name",
                match=models.MatchValue(value=decoded),
            ),
            models.FieldCondition(
                key="metadatas.search_type",
                match=models.MatchValue(value=search_type),
            ),
        ]
    )

    total_points = await qdrant_client.count(
        collection_name=FEEDBACK_COLLECTION,
        count_filter=query_filter,
    )
    total_pages = max(1, math.ceil(total_points.count / page_size))

    result = await qdrant_client.query_points(
        collection_name=FEEDBACK_COLLECTION,
        query_filter=query_filter,
        limit=page_size,
        offset=page_size * (page - 1),
    )

    return {
        "page": [point.payload for point in result.points],
        "page_info": {
            "total_elements": total_points.count,
            "total_pages": total_pages,
            "page": page,
            "first": page == 1,
            "last": page == total_pages,
            "empty": len(result.points) == 0,
        },
    }


@router.post(
    "/api/v1/feedback",
    operation_id="post_feedback",
    summary="피드백 저장",
)
async def post_feedback(
    collection_name: str = Body(..., description="대상 RAG 컬렉션 이름"),
    search_type: str = Body(..., description="검색 유형 (예: hybrid, dense)"),
    chat_id: str = Body(..., description="채팅 세션 ID"),
    query: str = Body(..., description="사용자 쿼리"),
    answer: str = Body(..., description="시스템 응답"),
):
    """
    쿼리를 dense 벡터로 임베딩하여 feedback_cache 컬렉션에 저장한다.

    포인트 payload 구조
    -------------------
    {
        "context": "<query>",
        "metadatas": {
            "chat_id": "...",
            "answer": "...",
            "collection_name": "...",
            "search_type": "..."
        }
    }
    """
    await _ensure_feedback_collection()

    # CPU 바운드 임베딩을 별도 스레드에서 실행하여 이벤트 루프 블로킹 방지
    dense_embedding = await asyncio.to_thread(
        embedding_model.encode, [query], normalize_embeddings=True
    )

    await qdrant_client.upsert(
        collection_name=FEEDBACK_COLLECTION,
        points=[
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector={"dense": dense_embedding[0].tolist()},
                payload={
                    "context": query,
                    "metadatas": {
                        "chat_id": chat_id,
                        "answer": answer,
                        "collection_name": collection_name,
                        "search_type": search_type,
                    },
                },
            )
        ],
    )
    logger.info(f"피드백 저장 완료: chat_id={chat_id}, collection={collection_name}")
    return {"result": "Feedback saved successfully."}


@router.delete(
    "/api/v1/feedback/{chat_id}",
    operation_id="delete_feedback",
    summary="피드백 삭제",
)
async def delete_feedback(chat_id: str):
    """chat_id에 해당하는 피드백 포인트를 feedback_cache에서 삭제한다."""
    is_exist = await qdrant_client.collection_exists(FEEDBACK_COLLECTION)
    if not is_exist:
        return JSONResponse(status_code=404, content={"result": "no data"})

    await qdrant_client.delete(
        collection_name=FEEDBACK_COLLECTION,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadatas.chat_id",
                        match=models.MatchValue(value=chat_id),
                    )
                ]
            )
        ),
    )
    logger.info(f"피드백 삭제 완료: chat_id={chat_id}")
    return {"result": "Feedback deleted successfully."}
