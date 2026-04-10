"""
routes/collection.py
====================
Qdrant 컬렉션 생명주기 및 문서 조회 엔드포인트.

담당 기능
---------
- 컬렉션 생성 / 조회 / 삭제
- 컬렉션 내 문서(포인트) 페이징 조회
- MongoDB에서 단락 원문 단건 조회
- MongoDB 인덱스 초기화 (create_documents_index)

의존 관계
---------
  qdrant_client  ← modules/dependencies.py
  embedding_model ← modules/dependencies.py (컬렉션 생성 시 벡터 차원 확인용)
  settings       ← config/__init__.py
"""

import html
import math
from logging import getLogger
from typing import Dict, List

from fastapi import APIRouter, Query
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel
from pymongo import ASCENDING
from qdrant_client.http import models

from config import settings
from modules.dependencies import get_embedding_model, get_qdrant_client

logger = getLogger(__name__)

router = APIRouter(tags=["Collection"])

# ── 모듈 레벨 싱글톤 (서버 기동 시 한 번만 생성) ────────────────────────────
qdrant_client = get_qdrant_client()
embedding_model = get_embedding_model()


# ════════════════════════════════════════════════════════════════════════════════
# Pydantic 모델
# ════════════════════════════════════════════════════════════════════════════════


class CreateCollection(BaseModel):
    """컬렉션 생성 요청 바디."""

    collection_name: str


class PageInfo(BaseModel):
    """페이징 메타데이터."""

    total_elements: int
    total_pages: int
    page: int
    first: bool  # 첫 페이지 여부
    last: bool  # 마지막 페이지 여부
    empty: bool  # 결과 없음 여부


class DocumentResponse(BaseModel):
    """문서 목록 페이징 응답."""

    page: List[Dict]
    page_info: PageInfo


# ════════════════════════════════════════════════════════════════════════════════
# 내부 헬퍼
# ════════════════════════════════════════════════════════════════════════════════


async def create_documents_index() -> None:
    """
    MongoDB rag-data.documents 컬렉션에 검색용 복합 인덱스를 생성한다.

    인덱스 구성: (collection_id, metadatas.doc_id, paragraph_id)
    - get_page() 호출 시마다 실행되지만, MongoDB는 중복 인덱스를 무시하므로 멱등하다.
    - 실패해도 서비스에 치명적이지 않으므로 예외를 삼킨다.
    """
    try:
        mongo_client = AsyncIOMotorClient(
            f"mongodb://{settings.db.host}:{settings.db.port}"
        )
        doc_collection = mongo_client["rag-data"]["documents"]
        await doc_collection.create_index(
            [
                ("collection_id", ASCENDING),
                ("metadatas.doc_id", ASCENDING),
                ("paragraph_id", ASCENDING),
            ]
        )
    except Exception as e:
        logger.warning(f"MongoDB 인덱스 생성 실패 (무시됨): {e}")


# ════════════════════════════════════════════════════════════════════════════════
# 컬렉션 CRUD
# ════════════════════════════════════════════════════════════════════════════════


@router.get("/api/v1/collection", operation_id="get_collection")
async def get_collections():
    """등록된 모든 Qdrant 컬렉션 목록을 반환한다."""
    result = await qdrant_client.get_collections()
    return result.collections


@router.post("/api/v1/collection", operation_id="create_collection")
async def create_collection(request: CreateCollection):
    """
    Qdrant 컬렉션을 생성한다. 이미 존재하면 아무 작업도 하지 않는다.

    벡터 설정
    ---------
    - dense  : COSINE 거리, HNSW (m=64, ef_construct=1000)
    - sparse : BM25 계열 Sparse 벡터 (on_disk=False → 메모리 상주)

    옵티마이저 설정
    ---------------
    - 세그먼트 10개, 최대 최적화 스레드 8개
    - 대용량 컬렉션을 고려해 인덱싱 임계값을 높게 설정
    """
    is_exist = await qdrant_client.collection_exists(
        collection_name=request.collection_name
    )
    if not is_exist:
        dimension = embedding_model.get_sentence_embedding_dimension()
        await qdrant_client.create_collection(
            collection_name=request.collection_name,
            vectors_config={
                "dense": models.VectorParams(
                    size=dimension,
                    distance=models.Distance.COSINE,
                    hnsw_config=models.HnswConfigDiff(
                        m=64, ef_construct=1000, full_scan_threshold=50000
                    ),
                )
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=False)
                )
            },
            optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=50000,
                memmap_threshold=100000,
                default_segment_number=10,
                max_optimization_threads=8,
            ),
        )
        logger.info(f"컬렉션 생성 완료: {request.collection_name} (dim={dimension})")

    return {"result": "Create collection successfully."}


@router.delete(
    "/api/v1/collection/{collection_name}",
    operation_id="delete_collection",
    description="collection_name은 HTML 인코딩된 문자열로 전달한다.",
)
async def delete_collection(collection_name: str):
    """Qdrant 컬렉션과 내부의 모든 포인트를 영구 삭제한다."""
    decoded = html.unescape(collection_name)
    await qdrant_client.delete_collection(decoded)
    logger.info(f"컬렉션 삭제 완료: {decoded}")
    return {"result": "Delete collection successfully."}


# ════════════════════════════════════════════════════════════════════════════════
# 문서 조회
# ════════════════════════════════════════════════════════════════════════════════


@router.get(
    "/api/v1/documents/{collection_name}",
    operation_id="get_all_documents",
    description="collection_name은 HTML 인코딩된 문자열로 전달한다.",
    response_model=DocumentResponse,
)
async def get_all_documents(
    collection_name: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=1000),
):
    """
    Qdrant 컬렉션의 포인트(문서 청크)를 페이징으로 반환한다.

    반환되는 page 배열의 각 원소는 포인트의 payload 딕셔너리이다.
    (벡터 값 제외)
    """
    decoded = html.unescape(collection_name)

    # 전체 포인트 수 조회 → 총 페이지 수 계산
    total_points = await qdrant_client.count(collection_name=decoded)
    total_pages = max(1, math.ceil(total_points.count / page_size))

    result = await qdrant_client.query_points(
        collection_name=decoded,
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


@router.get("/api/v1/{collection_name}/{p_id}/paragraph/content")
async def get_paragraph_content(collection_name: str, p_id: str):
    """
    MongoDB rag-data.documents에서 특정 단락(paragraph)의 원문을 단건 조회한다.

    Parameters
    ----------
    collection_name : str
        coolection_name (MongoDB 문서의 collection_id 필드)
    p_id : str
        paragraph_id (MongoDB 문서의 paragraph_id 필드)
    """
    mongo_client = AsyncIOMotorClient(
        f"mongodb://{settings.db.host}:{settings.db.port}"
    )
    try:
        page_collection = mongo_client["rag-data"]["documents"]
        document = await page_collection.find_one(
            {"collection_id": collection_name, "paragraph_id": p_id}
        )
        if document is None:
            return {}
        del document["_id"]
        return document
    finally:
        mongo_client.close()
