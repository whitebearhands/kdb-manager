"""
routes/document.py
==================
문서 임베딩 업서트, 삭제, MongoDB 저장 엔드포인트.

담당 기능
---------
- 문서(청크) 벡터 임베딩 후 Qdrant에 upsert
- 파일 단위 / 문서 ID 단위 포인트 삭제
- MongoDB에 단락(paragraph), 청크(chunk) 원문 저장
- 적응형 배치 처리 및 업서트 무결성 검증

처리 파이프라인
---------------
  요청 문서 리스트
      ↓
  adaptive_batch_upsert()  ← 동적 배치 크기 조절 (OOM 방지)
      ↓ [dense + sparse 병렬 임베딩]
  upsert_batch_with_retry()  ← 지수 백오프 재시도
      ↓
  verify_upsert_integrity()  ← 샘플링 무결성 검증
      ↓
  실패 문서 자동 재시도 (failed_indices < 50개)

의존 관계
---------
  embedding_model     ← modules/dependencies.py  (dense 임베딩)
  sparse_embedding_model ← modules/dependencies.py  (sparse 임베딩)
  qdrant_client       ← modules/dependencies.py
  get_chunk_id()      ← modules/dependencies.py  (포인트 ID 생성)
  create_collection() ← routes/collection.py     (없으면 컬렉션 먼저 생성)
  settings            ← config/__init__.py
"""

import asyncio
import gc
import html
import random
from logging import getLogger
from typing import Any, Dict, List, Optional, Tuple

import torch
from fastapi import APIRouter, Body
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel
from qdrant_client.http import models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from config import settings
from modules.dependencies import (
    get_chunk_id,
    get_embedding_model,
    get_model_device,
    get_qdrant_client,
    get_sparse_model,
)
from routes.collection import CreateCollection, create_collection, create_documents_index

logger = getLogger(__name__)

router = APIRouter(tags=["Document"])

# ── 모듈 레벨 싱글톤 ────────────────────────────────────────────────────────
embedding_model = get_embedding_model()
sparse_embedding_model = get_sparse_model()
qdrant_client = get_qdrant_client()


# ════════════════════════════════════════════════════════════════════════════════
# Pydantic 모델
# ════════════════════════════════════════════════════════════════════════════════


class Document(BaseModel):
    """단일 문서 청크."""
    context: str          # 청크 본문 텍스트
    ids: str              # 원본 문서 ID
    page_number: int = -1 # 원본 문서의 페이지 번호 (-1 = 알 수 없음)
    size: int             # 청크 글자 수
    metadatas: Dict[str, Any]  # 파일명, doc_id, collection_id 등 부가 정보


class UpsertDocument(BaseModel):
    """문서 업서트 요청 바디."""
    collection_name: str
    documents: List[Document]


# ════════════════════════════════════════════════════════════════════════════════
# 내부 헬퍼 — 메모리 관리
# ════════════════════════════════════════════════════════════════════════════════


def free_memory() -> None:
    """
    GC 실행 및 CUDA 캐시 비우기.
    대량 임베딩 직후 호출해 메모리 압박을 줄인다.
    """
    gc.collect()
    if get_model_device() == "cuda":
        torch.cuda.empty_cache()


# ════════════════════════════════════════════════════════════════════════════════
# 내부 헬퍼 — Sparse 벡터 생성
# ════════════════════════════════════════════════════════════════════════════════


def create_sparse_vector(text: str) -> models.SparseVector:
    """
    단일 텍스트를 BM25 계열 Sparse 벡터로 변환한다.

    Parameters
    ----------
    text : str
        인코딩할 텍스트

    Returns
    -------
    models.SparseVector
        Qdrant SparseVector (indices + values)

    Raises
    ------
    ValueError
        모델이 indices/values 속성을 반환하지 않는 경우
    """
    embeddings = list(sparse_embedding_model.embed([text]))[0]
    if hasattr(embeddings, "indices") and hasattr(embeddings, "values"):
        return models.SparseVector(
            indices=embeddings.indices.tolist(),
            values=embeddings.values.tolist(),
        )
    raise ValueError("Sparse 임베딩 모델이 indices/values를 반환하지 않습니다.")


def create_sparse_vectors_batch(texts: List[str]) -> List[models.SparseVector]:
    """
    텍스트 리스트를 배치로 Sparse 벡터화한다.

    단건 반복 호출보다 모델 내부 최적화를 활용할 수 있어 더 빠르다.

    Parameters
    ----------
    texts : List[str]
        인코딩할 텍스트 목록

    Returns
    -------
    List[models.SparseVector]
        texts와 동일한 순서의 SparseVector 목록

    Raises
    ------
    ValueError
        하나라도 indices/values 속성이 없으면 즉시 예외
    """
    results = []
    for embedding in sparse_embedding_model.embed(texts):
        if hasattr(embedding, "indices") and hasattr(embedding, "values"):
            results.append(
                models.SparseVector(
                    indices=embedding.indices.tolist(),
                    values=embedding.values.tolist(),
                )
            )
        else:
            raise ValueError("Sparse 임베딩 모델이 indices/values를 반환하지 않습니다.")
    return results


# ════════════════════════════════════════════════════════════════════════════════
# 내부 헬퍼 — 재시도 로직
# ════════════════════════════════════════════════════════════════════════════════


async def retry_with_backoff(
    func,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 30.0,
    backoff_factor: float = 2.0,
) -> Any:
    """
    지수 백오프(Exponential Backoff)로 비동기 함수를 재시도한다.

    Parameters
    ----------
    func : Callable[[], Awaitable]
        재시도할 인자 없는 비동기 함수 (lambda 또는 partial 권장)
    max_retries : int
        최대 시도 횟수
    initial_delay : float
        첫 재시도 대기 시간(초)
    max_delay : float
        대기 시간 상한(초)
    backoff_factor : float
        매 실패마다 대기 시간에 곱하는 배수

    Raises
    ------
    Exception
        max_retries 소진 후에도 실패하면 마지막 예외를 raise
    """
    delay = initial_delay
    last_exception: Optional[Exception] = None

    for attempt in range(max_retries):
        try:
            return await func()
        except (ResponseHandlingException, UnexpectedResponse, ConnectionError) as e:
            last_exception = e
            if attempt < max_retries - 1:
                logger.warning(f"재시도 {attempt + 1}/{max_retries} ({delay:.1f}s 후): {e}")
                await asyncio.sleep(delay)
                delay = min(delay * backoff_factor, max_delay)
            else:
                logger.error(f"{max_retries}회 시도 후 실패: {e}")
                raise
        except Exception:
            raise

    if last_exception:
        raise last_exception


async def upsert_batch_with_retry(
    collection_name: str,
    points: List[models.PointStruct],
    max_retries: int = 3,
) -> Tuple[bool, Optional[Any]]:
    """
    포인트 배치를 재시도 로직과 함께 Qdrant에 upsert한다.

    Returns
    -------
    (성공 여부, upsert 결과 또는 None)
    """
    async def _upsert():
        return await qdrant_client.upsert(
            collection_name=collection_name, points=points, wait=True
        )

    try:
        result = await retry_with_backoff(_upsert, max_retries=max_retries)
        return True, result
    except Exception as e:
        logger.error(f"배치 upsert 최종 실패: {e}")
        return False, None


# ════════════════════════════════════════════════════════════════════════════════
# 내부 헬퍼 — 적응형 배치 업서트
# ════════════════════════════════════════════════════════════════════════════════


async def adaptive_batch_upsert(
    collection_name: str,
    documents: List[Document],
    initial_batch_size: int = 100,
    min_batch_size: int = 10,
) -> Dict[str, Any]:
    """
    메모리 상황에 따라 배치 크기를 동적으로 조절하며 문서를 upsert한다.

    동작 원리
    ---------
    1. initial_batch_size로 시작
    2. upsert 성공 시 → 배치 크기 유지 (최대 initial_batch_size)
    3. upsert 실패 시 → 배치 크기 절반으로 감소 (min_batch_size 하한)
    4. min_batch_size에서도 실패 → 해당 배치 건너뛰고 failed_indices에 기록
    5. MemoryError 발생 시 → 배치 크기 즉시 절반으로 감소, 2초 대기 후 재시도

    임베딩 전략
    -----------
    - dense + sparse를 asyncio.gather로 병렬 생성 (CPU-bound이므로 to_thread 사용)
    - 10KB 초과 텍스트는 잘라냄 (임베딩 모델 토큰 한계 대비)

    Returns
    -------
    {
        "successful_count": int,     성공한 포인트 수
        "failed_count": int,         실패한 문서 수
        "failed_indices": List[int], 실패한 원본 리스트 인덱스
        "checkpoint_data": List      무결성 검증용 포인트 ID 목록
    }
    """
    batch_size = initial_batch_size
    successful_count = 0
    failed_indices: List[int] = []
    checkpoint_data: List[Dict] = []

    total_docs = len(documents)
    i = 0

    while i < total_docs:
        batch_end = min(i + batch_size, total_docs)
        batch_docs = documents[i:batch_end]

        try:
            gc.collect()

            # 10KB 초과 텍스트 잘라내기
            for doc in batch_docs:
                if len(doc.context) > 10000:
                    logger.warning(
                        f"문서 텍스트 초과 ({len(doc.context)}자) → 10000자로 잘라냄"
                    )
                    doc.context = doc.context[:10000]

            sentence_pairs = [doc.context for doc in batch_docs]

            # Dense + Sparse 임베딩 병렬 생성
            # - embedding_model.encode: CPU-bound → asyncio.to_thread로 이벤트 루프 블로킹 방지
            # - create_sparse_vectors_batch: 동일 이유로 to_thread 사용
            dense_embedding, sparse_vectors = await asyncio.gather(
                asyncio.to_thread(
                    embedding_model.encode,
                    sentence_pairs,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                ),
                asyncio.to_thread(create_sparse_vectors_batch, sentence_pairs),
            )

            # Qdrant 포인트 구성
            points = []
            for index, doc in enumerate(batch_docs):
                point_id = get_chunk_id()
                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector={
                            "dense": dense_embedding[index].tolist(),
                            "sparse": sparse_vectors[index],
                        },
                        payload={
                            "context": doc.context,
                            "ids": doc.ids,
                            "page_number": doc.page_number,
                            "size": doc.size,
                            "metadatas": doc.metadatas,
                        },
                    )
                )
                checkpoint_data.append(
                    {"doc_index": i + index, "point_id": point_id, "doc_id": doc.ids}
                )

            success, _ = await upsert_batch_with_retry(collection_name, points)

            if success:
                successful_count += len(points)
                logger.info(f"배치 {i}~{batch_end} 성공 ({len(points)}건)")
                i = batch_end
            else:
                # 실패 → 배치 크기 절반 감소 후 재시도
                if batch_size > min_batch_size:
                    batch_size = max(batch_size // 2, min_batch_size)
                    logger.warning(f"배치 크기 감소 → {batch_size}")
                else:
                    # 최소 크기에서도 실패하면 건너뜀
                    failed_indices.extend(range(i, batch_end))
                    logger.error(f"배치 {i}~{batch_end} 최종 실패 → 건너뜀")
                    i = batch_end

        except MemoryError:
            logger.error(f"OOM 발생 (batch_size={batch_size})")
            if batch_size <= min_batch_size:
                # 최소 크기에서 OOM → 해당 배치 포기
                logger.critical(f"배치 {i}~{batch_end} OOM으로 건너뜀")
                failed_indices.extend(range(i, batch_end))
                i = batch_end
            else:
                # 배치 크기 절반으로 줄이고 재시도
                batch_size = max(batch_size // 2, min_batch_size)
                logger.warning(f"OOM으로 배치 크기 감소 → {batch_size}")
            gc.collect()
            await asyncio.sleep(2)

        except Exception as e:
            logger.error(f"배치 {i}~{batch_end} 예외: {e}")
            failed_indices.extend(range(i, batch_end))
            i = batch_end

    return {
        "successful_count": successful_count,
        "failed_count": len(failed_indices),
        "failed_indices": failed_indices,
        "checkpoint_data": checkpoint_data,
    }


# ════════════════════════════════════════════════════════════════════════════════
# 내부 헬퍼 — 업서트 무결성 검증
# ════════════════════════════════════════════════════════════════════════════════


async def verify_upsert_integrity(
    collection_name: str,
    checkpoint_data: List[Dict],
    sample_rate: float = 0.1,
) -> Dict[str, Any]:
    """
    upsert 후 포인트가 실제로 저장됐는지 샘플링으로 검증한다.

    Parameters
    ----------
    checkpoint_data : List[Dict]
        adaptive_batch_upsert가 반환한 checkpoint 목록
    sample_rate : float
        검증할 포인트의 비율 (기본 10%)

    Returns
    -------
    {
        "verified": bool,         전체 샘플의 95% 이상 확인 시 True
        "integrity_rate": float,  실제 확인된 비율
        "sample_size": int,
        "missing_count": int,
        "missing_ids": List       최대 10개의 누락 포인트 ID
    }
    """
    if not checkpoint_data:
        return {"verified": True, "message": "검증할 데이터가 없습니다."}

    sample_size = max(1, int(len(checkpoint_data) * sample_rate))
    sample_indices = random.sample(range(len(checkpoint_data)), sample_size)

    verified_count = 0
    missing_ids: List = []

    for idx in sample_indices:
        point_id = checkpoint_data[idx]["point_id"]
        try:
            result = await qdrant_client.retrieve(
                collection_name=collection_name,
                ids=[point_id],
                with_payload=False,
                with_vectors=False,
            )
            if result:
                verified_count += 1
            else:
                missing_ids.append(point_id)
        except Exception as e:
            logger.error(f"포인트 {point_id} 검증 실패: {e}")
            missing_ids.append(point_id)

    integrity_rate = verified_count / sample_size if sample_size > 0 else 0

    return {
        "verified": integrity_rate >= 0.95,  # 95% 이상이면 정상으로 간주
        "integrity_rate": integrity_rate,
        "sample_size": sample_size,
        "missing_count": len(missing_ids),
        "missing_ids": missing_ids[:10],
    }


# ════════════════════════════════════════════════════════════════════════════════
# 내부 헬퍼 — MongoDB 저장
# ════════════════════════════════════════════════════════════════════════════════


async def paragraph_to_mongo(paragraph_list: List[Dict]) -> None:
    """
    단락(paragraph) 목록을 MongoDB rag-data.documents 컬렉션에 저장한다.

    저장 전 검색용 인덱스가 없으면 자동 생성한다.
    """
    if not paragraph_list:
        return
    await create_documents_index()
    mongo_client = AsyncIOMotorClient(
        f"mongodb://{settings.db.host}:{settings.db.port}"
    )
    try:
        doc_collection = mongo_client["rag-data"]["documents"]
        await doc_collection.insert_many(paragraph_list)
        logger.info(f"단락 {len(paragraph_list)}건 MongoDB 저장 완료")
    finally:
        mongo_client.close()


async def chunk_to_mongo(collection_name: str, chunks: List[Dict]) -> None:
    """
    청크(chunk) 목록을 MongoDB chunks.<collection_name> 컬렉션에 저장한다.

    배치 처리 (100건) + 지수 백오프 재시도 (최대 3회)를 적용한다.
    MongoDB의 ordered=False를 사용해 부분 실패를 허용한다.
    """
    if not chunks:
        return

    mongo_client = AsyncIOMotorClient(
        f"mongodb://{settings.db.host}:{settings.db.port}"
    )
    try:
        doc_collection = mongo_client["chunks"][collection_name]

        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            retry_count = 3
            while retry_count > 0:
                try:
                    await doc_collection.insert_many(batch, ordered=False)
                    break
                except Exception as e:
                    retry_count -= 1
                    if retry_count == 0:
                        logger.error(f"청크 배치 {i}~{i+batch_size} 저장 실패: {e}")
                    else:
                        # 1회 실패 → 2초, 2회 실패 → 4초 대기
                        await asyncio.sleep(2 ** (3 - retry_count))
    finally:
        mongo_client.close()


# ════════════════════════════════════════════════════════════════════════════════
# 엔드포인트
# ════════════════════════════════════════════════════════════════════════════════


@router.post("/api/v1/documents", operation_id="upsert_documents")
async def upsert_documents(request: UpsertDocument):
    """
    문서 청크 목록을 임베딩 후 Qdrant에 upsert한다.

    처리 순서
    ---------
    1. 컬렉션이 없으면 자동 생성
    2. adaptive_batch_upsert로 전체 문서 upsert (동적 배치, 재시도 포함)
    3. 샘플링으로 무결성 검증 (10% 샘플, 95% 이상 확인 시 통과)
    4. 실패 문서가 50건 미만이면 소규모 배치(30건)로 자동 재시도
    5. 메모리 정리 (GC + CUDA 캐시)

    응답 예시
    ---------
    {
        "result": "95/100 items added successfully",
        "details": {
            "total_requested": 100,
            "successful": 95,
            "failed": 5,
            "integrity_verified": true,
            "integrity_rate": 0.97
        }
    }
    """
    try:
        # 컬렉션 없으면 자동 생성
        await create_collection(
            CreateCollection(collection_name=request.collection_name)
        )

        total = len(request.documents)
        logger.info(f"[upsert] 시작: {request.collection_name} / {total}건")

        # 1차 적응형 업서트
        result = await adaptive_batch_upsert(
            collection_name=request.collection_name,
            documents=request.documents,
            initial_batch_size=100,
            min_batch_size=5,
        )

        # 무결성 샘플링 검증
        verification = await verify_upsert_integrity(
            collection_name=request.collection_name,
            checkpoint_data=result["checkpoint_data"],
            sample_rate=0.1,
        )

        # 실패 문서 자동 재시도 (소규모만)
        retry_count = 0
        if result["failed_indices"] and len(result["failed_indices"]) < 50:
            logger.info(f"[upsert] 실패 {len(result['failed_indices'])}건 재시도")
            failed_docs = [request.documents[i] for i in result["failed_indices"]]
            retry_result = await adaptive_batch_upsert(
                collection_name=request.collection_name,
                documents=failed_docs,
                initial_batch_size=30,
                min_batch_size=1,
            )
            retry_count = retry_result["successful_count"]

        final_success = result["successful_count"] + retry_count
        final_failed = result["failed_count"] - retry_count

        response: Dict[str, Any] = {
            "result": f"{final_success}/{total} items added successfully",
            "details": {
                "total_requested": total,
                "successful": final_success,
                "failed": final_failed,
                "integrity_verified": verification["verified"],
                "integrity_rate": verification.get("integrity_rate", 0),
            },
        }
        if final_failed > 0:
            response["warning"] = f"{final_failed}건 upsert 실패"
            logger.warning(f"[upsert] 최종 실패: {final_failed}건")

        logger.info(f"[upsert] 완료: 성공 {final_success}/{total}건")
        return response

    finally:
        free_memory()


@router.post("/api/v1/mongo/paragraph", operation_id="update_paragraph_to_mongo")
async def update_paragraph_to_mongo(pages=Body(...)):
    """
    단락 목록을 MongoDB rag-data.documents에 저장한다.

    Request body: {"pages": [...paragraph dicts...]}

    RabbitMQ 파이프라인 외부에서 직접 단락을 저장할 때 사용한다.
    """
    await paragraph_to_mongo(pages["pages"])
    return {"status": "ok"}


@router.delete("/api/v1/file", operation_id="remove_file")
async def remove_file(
    file_name: str = Body(...),
    collection_name: str = Body(...),
):
    """
    파일 이름으로 해당 파일의 모든 청크 포인트를 Qdrant에서 삭제한다.

    metadatas.file_name 필드를 기준으로 필터링한다.
    """
    result = await qdrant_client.delete(
        collection_name,
        points_selector=models.Filter(
            must=[
                models.FieldCondition(
                    key="metadatas.file_name",
                    match=models.MatchValue(value=file_name),
                )
            ]
        ),
        wait=True,
    )
    logger.info(f"파일 삭제 완료: {file_name} / {collection_name}")
    return {"result": result.model_dump()}


@router.delete(
    "/api/v1/documents/{collection_name}/{id}",
    operation_id="remove_document",
    description="collection_name은 HTML 인코딩된 문자열로 전달한다.",
)
async def remove_document(collection_name: str, id: str):
    """
    문서 ID(ids 필드)로 해당 포인트를 Qdrant에서 삭제한다.

    collection_name은 URL 인코딩 상태로 전달되며, 내부에서 디코딩한다.
    """
    decoded = html.unescape(collection_name)
    result = await qdrant_client.delete(
        decoded,
        points_selector=models.Filter(
            must=[
                models.FieldCondition(key="id", match=models.MatchValue(value=id))
            ]
        ),
    )
    return {"result": result.model_dump()}
