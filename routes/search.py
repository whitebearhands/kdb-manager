"""
routes/search.py
================
하이브리드 벡터 검색, 리랭킹, 단락 복원 엔드포인트.

검색 파이프라인
---------------
  1. 쿼리 임베딩 (dense + sparse 동시)
  2. Qdrant Prefetch → RRF 퓨전  (하이브리드 검색)
  3. 외부 리랭커 호출              (get_rerank)
  4. MongoDB에서 단락 원문 조회    (get_page, use_paragraph=True 시)

엔드포인트 요약
---------------
  POST /api/v2/document/search_rerank   하이브리드 검색 + 리랭킹 (메인)
  POST /api/v2/document/search_paragraph 검색 + 리랭킹 후 paragraph_id/score만 반환
  POST /api/v1/document/search          리랭킹 없는 raw 하이브리드 검색

의존 관계
---------
  embedding_model      ← modules/dependencies.py
  sparse_embedding_model ← modules/dependencies.py
  qdrant_client        ← modules/dependencies.py
  create_documents_index ← routes/collection.py
  settings             ← config/__init__.py
"""

import json
from logging import getLogger
from typing import Any, Dict, List, Optional

import aiohttp
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from motor.motor_asyncio import AsyncIOMotorClient
from qdrant_client.http import models

from config import settings
from modules.dependencies import get_embedding_model, get_qdrant_client, get_sparse_model
from routes.collection import create_documents_index

logger = getLogger(__name__)

router = APIRouter(tags=["Search"])

# ── 모듈 레벨 싱글톤 ────────────────────────────────────────────────────────
embedding_model = get_embedding_model()
sparse_embedding_model = get_sparse_model()
qdrant_client = get_qdrant_client()


# ════════════════════════════════════════════════════════════════════════════════
# Pydantic 모델
# ════════════════════════════════════════════════════════════════════════════════


from pydantic import BaseModel


class SearchDocument(BaseModel):
    """검색 요청 바디."""
    collection_name: str
    query: str

    # 메타데이터 필터 (metadatas.<key> IN match_values)
    metadata_filter_key: str | None = None
    match_values: List[str] | None = None

    # room_id 필터 — 채팅방 단위 검색 범위 제한
    room_id: Optional[str] = None

    top_k: int = 100           # Qdrant prefetch 및 fusion 결과 상한
    use_paragraph: bool = False  # True면 MongoDB에서 단락 원문 조회 후 반환


# ════════════════════════════════════════════════════════════════════════════════
# 내부 헬퍼 — 하이브리드 검색
# ════════════════════════════════════════════════════════════════════════════════


async def get_search_result(
    collection_name: str,
    query: str,
    metadata_filter_key: Optional[str],
    match_values: Optional[List[str]],
    top_k: int,
    room_id: Optional[str] = None,
) -> models.QueryResponse:
    """
    Dense + Sparse 하이브리드 검색을 수행하고 RRF 퓨전 결과를 반환한다.

    검색 구성
    ---------
    - Prefetch: sparse(BM25) top_k + dense(COSINE) top_k
    - Fusion:   RRF (Reciprocal Rank Fusion) — 두 결과를 점수 기반으로 통합
    - Filter:   room_id (선택), metadata_filter_key/match_values (선택)
    - limit:    50 (리랭커 입력으로 충분한 후보군)

    Parameters
    ----------
    room_id : str | None
        metadatas.room_id 필터. 채팅방별 문서 격리에 사용.
    metadata_filter_key : str | None
        metadatas 하위 필드명 (예: "doc_id", "file_name")
    match_values : List[str] | None
        해당 필드가 match_values 중 하나와 일치하는 포인트만 검색
    """
    dense_query = embedding_model.encode(query, normalize_embeddings=True)
    sparse_query = list(sparse_embedding_model.embed([query]))[0]

    prefetch = [
        models.Prefetch(
            query=models.SparseVector(
                indices=sparse_query.indices.tolist(),
                values=sparse_query.values.tolist(),
            ),
            using="sparse",
            limit=top_k,
        ),
        models.Prefetch(query=dense_query, using="dense", limit=top_k),
    ]

    # 필터 조건 조합 (room_id + 메타데이터 키)
    conditions = []
    if room_id:
        conditions.append(
            models.FieldCondition(
                key="metadatas.room_id",
                match=models.MatchAny(any=[room_id]),
            )
        )
    if metadata_filter_key and match_values:
        conditions.append(
            models.FieldCondition(
                key=f"metadatas.{metadata_filter_key}",
                match=models.MatchAny(any=match_values),
            )
        )

    query_filter = models.Filter(must=conditions) if conditions else None

    fusion_results = await qdrant_client.query_points(
        collection_name=collection_name,
        prefetch=prefetch,
        query_filter=query_filter,
        search_params=models.SearchParams(hnsw_ef=128),
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=50,  # 리랭커에 넘길 후보 수 (top_k보다 넉넉히)
    )

    return fusion_results


# ════════════════════════════════════════════════════════════════════════════════
# 내부 헬퍼 — 외부 리랭커
# ════════════════════════════════════════════════════════════════════════════════


async def get_rerank(
    query: str,
    fusion: List[Any],
    rerank_top_k: int = 5,
) -> List[Dict]:
    """
    외부 리랭커 서비스에 HTTP POST로 청크 재순위 요청을 보낸다.

    요청 형식 (settings.a2o.reranker/v2/predict)
    -------------------------------------------
    {
        "reqid": "...",
        "input": {
            "items": [{
                "query": "...",
                "k": 5,
                "retrieved_chunks": [{"context": ..., "ids": ..., "metadatas": ...}]
            }]
        }
    }

    실패 시 빈 리스트를 반환하며 서비스 전체를 중단시키지 않는다.
    (리랭커 장애 시 검색 자체가 실패하는 것보다 빈 결과가 낫다는 판단)
    """
    url = f"{settings.a2o.reranker}/v2/predict"

    # Qdrant 포인트 → 리랭커 입력 형식으로 변환
    chunks = [
        {
            "context": point.payload["context"],
            "ids": point.payload["ids"],
            "metadatas": point.payload["metadatas"],
        }
        for point in fusion
    ]

    packet = {
        "reqid": "kdb-manager",
        "input": {
            "items": [{"query": query, "k": rerank_top_k, "retrieved_chunks": chunks}]
        },
    }

    try:
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                url,
                json=packet,
                headers={"Content-Type": "application/json"},
            ) as response:
                res_json = await response.json(content_type=None)
                return res_json["output"]["items"][0]["top_k_chunks"]
    except Exception as e:
        logger.error(f"리랭커 호출 실패 ({url}): {e}")
        return []


# ════════════════════════════════════════════════════════════════════════════════
# 내부 헬퍼 — MongoDB 단락 복원
# ════════════════════════════════════════════════════════════════════════════════


async def get_page(chunks: List[Dict]) -> List[Dict]:
    """
    리랭킹된 청크 목록을 받아 MongoDB에서 단락 원문을 조회하고 합친다.

    동작 원리
    ---------
    - 각 청크의 metadatas에서 doc_id, paragraph_id, collection_id를 읽어
      MongoDB rag-data.documents 컬렉션에서 원문 단락을 조회한다.
    - paragraph_type == "faq"인 청크는 DB 조회 없이 그대로 반환한다.
    - 동일 paragraph_id가 여러 청크에서 등장할 경우 첫 번째만 포함한다.
      (중복 제거 — added_p_ids set 활용)
    - DB에서 단락을 찾지 못하면 원본 청크를 그대로 반환한다.

    입력 형식
    ---------
    chunks 원소는 리랭커가 반환한 dict이거나 (tuple인 경우 (chunk, score) 쌍).

    Returns
    -------
    단락 원문이 포함된 dict 목록.
    각 원소에 rerank_score 필드가 추가된다.
    """
    await create_documents_index()

    mongo_client = AsyncIOMotorClient(
        f"mongodb://{settings.db.host}:{settings.db.port}"
    )

    try:
        page_collection = mongo_client["rag-data"]["documents"]
        result: List[Dict] = []
        added_p_ids: set = set()  # 중복 단락 방지

        # chunks가 (chunk, score) 튜플 형식인 경우와 dict 형식 모두 처리
        is_tuple_format = chunks and isinstance(chunks[0], tuple)

        items = chunks if is_tuple_format else [(chunk, None) for chunk in chunks]

        for item in items:
            if is_tuple_format:
                chunk, score = item
            else:
                chunk, score = item

            # faq 타입은 DB 조회 없이 그대로 반환
            if chunk.get("metadatas", {}).get("paragraph_type") == "faq":
                result.append(chunk)
                continue

            doc_id = chunk.get("metadatas", {}).get("doc_id")
            paragraph_id = chunk.get("metadatas", {}).get("paragraph_id")
            collection_id = chunk.get("metadatas", {}).get("collection_id")

            # 식별자가 없는 청크는 원본 그대로 반환
            if not doc_id or not paragraph_id:
                result.append(chunk)
                continue

            page = await page_collection.find_one(
                {
                    "collection_id": collection_id,
                    "metadatas.doc_id": doc_id,
                    "paragraph_id": paragraph_id,
                }
            )

            if page:
                p_id = page.get("paragraph_id")
                if p_id and p_id in added_p_ids:
                    continue  # 중복 단락 건너뜀

                page["id"] = str(page["_id"])
                del page["_id"]

                # 리랭크 점수 부여 (입력 형식에 따라 필드 위치가 다름)
                if score is not None:
                    page["rerank_score"] = float(score)
                elif "reranked_score" in chunk:
                    page["rerank_score"] = float(chunk["reranked_score"])

                # bbox는 원본 청크의 값을 우선 사용
                if "bbox" in chunk.get("metadatas", {}):
                    page.setdefault("metadatas", {})["bbox"] = chunk["metadatas"]["bbox"]

                result.append(page)
                if p_id:
                    added_p_ids.add(p_id)
            else:
                # DB에서 단락을 찾지 못한 경우 원본 청크 반환
                result.append(chunk)

        return result

    finally:
        mongo_client.close()


# ════════════════════════════════════════════════════════════════════════════════
# 엔드포인트
# ════════════════════════════════════════════════════════════════════════════════


import time

from routes.document import free_memory


@router.post(
    "/api/v2/document/search_rerank",
    operation_id="search_rerank",
)
async def search_rerank(request: SearchDocument):
    """
    하이브리드 검색 → 리랭킹 → (선택) 단락 복원의 메인 검색 엔드포인트.

    use_paragraph=True 시 MongoDB에서 단락 원문을 조회해 반환하고,
    False면 리랭커가 반환한 청크 목록을 그대로 반환한다.

    room_id가 있으면 해당 채팅방의 문서로 검색 범위를 좁힌다.
    """
    start_time = time.time()
    try:
        fusion_results = await get_search_result(
            collection_name=request.collection_name,
            query=request.query,
            metadata_filter_key=request.metadata_filter_key,
            match_values=request.match_values,
            top_k=request.top_k,
            room_id=request.room_id,
        )

        reranked = await get_rerank(request.query, fusion_results.points, rerank_top_k=5)

        if request.use_paragraph:
            pages = await get_page(reranked)
            return JSONResponse(content=pages)
        return JSONResponse(content=reranked)

    except Exception as e:
        logger.exception(f"search_rerank 오류: {e}")
        return JSONResponse(status_code=500, content=str(e))
    finally:
        logger.info(f"[search_rerank] {request.query!r} → {time.time() - start_time:.3f}s")
        free_memory()


@router.post(
    "/api/v2/document/search_paragraph",
    operation_id="search_paragraph",
)
async def search_paragraph(request: SearchDocument):
    """
    하이브리드 검색 + 리랭킹 후 paragraph_id와 rerank_score만 반환한다.

    전체 단락 내용 없이 ID/점수만 필요한 경우 (예: 프론트에서 lazy-load) 사용한다.

    반환 예시
    ---------
    [{"paragraph_id": "abc", "score": 0.87}, ...]
    """
    start_time = time.time()
    try:
        fusion_results = await get_search_result(
            collection_name=request.collection_name,
            query=request.query,
            metadata_filter_key=request.metadata_filter_key,
            match_values=request.match_values,
            top_k=request.top_k,
        )

        reranked = await get_rerank(request.query, fusion_results.points, rerank_top_k=10)
        pages = await get_page(reranked)

        # 응답 축소: paragraph_id + score만 추출
        result = [
            {"paragraph_id": p["paragraph_id"], "score": p.get("rerank_score", 0)}
            for p in pages
        ]
        return JSONResponse(content=result)

    except Exception as e:
        logger.exception(f"search_paragraph 오류: {e}")
        return JSONResponse(status_code=500, content=str(e))
    finally:
        logger.info(f"[search_paragraph] {request.query!r} → {time.time() - start_time:.3f}s")
        free_memory()


@router.post("/api/v1/document/search", operation_id="search_document")
async def search_document(request: SearchDocument):
    """
    리랭킹 없는 순수 하이브리드 검색. Qdrant RRF 퓨전 결과를 그대로 반환한다.

    용도
    ----
    - 리랭커가 없는 환경
    - 빠른 1차 검색 결과가 필요한 경우
    - 리랭커 디버깅 시 fusion 결과와 비교

    score_threshold=0.11 이하의 포인트는 필터링된다.
    """
    dense_query = embedding_model.encode(request.query, normalize_embeddings=True)
    sparse_query = list(sparse_embedding_model.embed([request.query]))[0]

    prefetch = [
        models.Prefetch(
            query=models.SparseVector(
                indices=sparse_query.indices.tolist(),
                values=sparse_query.values.tolist(),
            ),
            using="sparse",
            limit=request.top_k,
        ),
        models.Prefetch(query=dense_query, using="dense", limit=request.top_k),
    ]

    query_filter = None
    if request.metadata_filter_key and request.match_values:
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key=f"metadatas.{request.metadata_filter_key}",
                    match=models.MatchAny(any=request.match_values),
                )
            ]
        )

    fusion_results = await qdrant_client.query_points(
        collection_name=request.collection_name,
        prefetch=prefetch,
        query_filter=query_filter,
        search_params=models.SearchParams(hnsw_ef=128),
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=request.top_k,
        score_threshold=0.11,
    )
    return fusion_results.points
