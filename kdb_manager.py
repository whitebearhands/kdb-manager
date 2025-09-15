# --- START OF FILE kdb_manager.py ---

import argparse
import contextlib
import html
import json
import math
import os
import random
import time
import uuid
import asyncio
from fastapi import Body, Depends, FastAPI, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from gridone_eureka_client import EurekaClient
from gridone_rabbitmq import ConnectionConfig, MessageContext, RabbitMQClient
import numpy as np
from pydantic import BaseModel
from pymongo import ASCENDING
import requests
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from modules.dependencies import (
    get_embedding_model,
    get_qdrant_client,
    get_chunk_id,
    get_sparse_model,
)
from query_normalizer import query_normalizer
from motor.motor_asyncio import AsyncIOMotorClient
from config import settings
from typing import Any, Dict, List, Union, Optional, Tuple
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from sentence_transformers import SentenceTransformer
from gridone_logger import setup_logger
from logging import getLogger
from gridone_redis import Redis
from gridone_rabbitmq.rag import RagProducer, RagEmbeddingWorker
from routes.feedback import router as FeedBackRouter

setup_logger(settings.log)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--port", help="config 파일의 위치. 파일명까지 경로를 지정해야 한다."
)
args, unknown = parser.parse_known_args()
if args.port:
    settings.app.port = int(args.port)

logger = getLogger(__name__)


eureka_client = EurekaClient()
handler_name = "kdb-manager"


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    config = {
        "port": settings.app.port,
        "service_name": settings.app.service_name,
        "eureka_server": settings.app.eureka_server,
        "use_ssl": False,
        "use_management": True,
        "site_key": "production",
    }
    # Eureka 등록 (actuator는 이미 설정됨)
    await eureka_client.run(app, config)
    # await setup_redis()
    # await setup_mq()

    yield  # 애플리케이션 실행

    # Shutdown
    await eureka_client.shutdown()
    # await clear_mq()
    # await clear_redis()


logger = getLogger(__name__)

# FastAPI 앱 인스턴스 생성 시 lifespan 핸들러를 등록합니다.
app = FastAPI(title="KDB SERVER", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(
    FeedBackRouter,
)

embedding_model = get_embedding_model()
sparse_embedding_model = get_sparse_model()
qdrant_client = get_qdrant_client()
# tokenizer = AutoTokenizer.from_pretrained(
#     "Qwen/Qwen3-Reranker-4B", trust_remote_code=True
# )
# if tokenizer.pad_token is None:
#     eos = tokenizer.eos_token or tokenizer.unk_token
#     # 방법 1) pad_token_id를 eos로 매핑(임베딩 리사이즈 X)
#     tokenizer.pad_token = eos

# model = AutoModelForSequenceClassification.from_pretrained(
#     "Qwen/Qwen3-Reranker-4B", trust_remote_code=True
# )
# model.config.pad_token_id = tokenizer.pad_token_id
# model.eval()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# if torch.backends.mps.is_available():
#     device = "mps"
# elif torch.cuda.is_available():
#     device = "cude"
# else:
#     device = "cpu"

# model.to(device)


def create_documents_index():
    mongo_client = AsyncIOMotorClient(
        f"mongodb://{settings.db.host}:{settings.db.port}"
    )
    doc_collection = mongo_client["rag-data"]["documents"]
    doc_collection.create_index(
        [
            ("collection_id", ASCENDING),
            ("metadatas.doc_id", ASCENDING),
            ("paragraph_id", ASCENDING),
        ]
    )


@app.delete(
    "/api/v1/collection/{collection_name}",
    operation_id="delete_collection",
    description="collection_name : html encoded string",
)
async def delete_collection(collection_name: str):
    decoded_collection_name = html.unescape(collection_name)
    await qdrant_client.delete_collection(decoded_collection_name)
    return {"result": "Delete collection successfully."}


# ... (이하 모든 기존 엔드포인트 코드는 그대로 유지) ...
@app.get("/api/v1/collection", operation_id="get_collection")
async def get_collections():
    result = await qdrant_client.get_collections()
    return result.collections


class CreateCollection(BaseModel):
    collection_name: str


@app.post("/api/v1/collection", operation_id="create_collection")
async def create_collection(request: CreateCollection):
    is_exist = await qdrant_client.collection_exists(
        collection_name=f"{request.collection_name}"
    )
    if not is_exist:
        await qdrant_client.create_collection(
            collection_name=request.collection_name,
            vectors_config={
                "dense": models.VectorParams(
                    size=embedding_model.get_sentence_embedding_dimension(),
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

        await qdrant_client.create_payload_index(
            collection_name=request.collection_name,
            field_name="metadatas.class",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )

    return {"result": "Create collection successfully."}


class PageInfo(BaseModel):
    total_elements: int
    total_pages: int
    page: int
    first: int
    last: int
    empty: bool


class DocumentResponse(BaseModel):
    page: List[Dict]
    page_info: PageInfo


@app.get(
    "/api/v1/documents/{collection_name}",
    operation_id="get_all_documents",
    description="collection_name : html encoded string",
    response_model=DocumentResponse,
)
async def get_all_documents(
    collection_name: str,
    page: int = Query(1),
    page_size: int = Query(10),
):
    decoded_collection_name = html.unescape(collection_name)
    total_points = await qdrant_client.count(collection_name=decoded_collection_name)
    total_pages = max(1, math.ceil(total_points.count / page_size))
    result = await qdrant_client.query_points(
        collection_name=decoded_collection_name,
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


class Document(BaseModel):
    context: str
    ids: str
    page_number: int = -1
    size: int
    metadatas: Dict[str, Union[str, int, float, list, dict]]


class UpsertDocument(BaseModel):
    collection_name: str
    documents: List[Document]


async def retry_with_backoff(
    func,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 30.0,
    backoff_factor: float = 2.0,
):
    """재시도 로직 with exponential backoff"""
    delay = initial_delay
    last_exception = None

    for attempt in range(max_retries):
        try:
            return await func()
        except (ResponseHandlingException, UnexpectedResponse, ConnectionError) as e:
            last_exception = e
            if attempt < max_retries - 1:
                logger.warning(
                    f"Retry {attempt + 1}/{max_retries} after {delay}s: {str(e)}"
                )
                await asyncio.sleep(delay)
                delay = min(delay * backoff_factor, max_delay)
            else:
                logger.error(f"Failed after {max_retries} attempts: {str(e)}")
                raise
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise

    if last_exception:
        raise last_exception


async def upsert_batch_with_retry(
    collection_name: str,
    points: List[models.PointStruct],
    max_retries: int = 3,
) -> Tuple[bool, Optional[Any]]:
    """배치 upsert with retry"""

    async def _upsert():
        return await qdrant_client.upsert(
            collection_name=collection_name, points=points, wait=True
        )

    try:
        result = await retry_with_backoff(_upsert, max_retries=max_retries)
        return True, result
    except Exception as e:
        logger.error(f"Batch upsert failed: {str(e)}")
        return False, None


async def adaptive_batch_upsert(
    collection_name: str,
    documents: List[Document],
    initial_batch_size: int = 100,  # 기존 크기부터 시작
    min_batch_size: int = 10,  # 문제 시에만 1개씩
) -> Dict[str, Any]:
    """동적 배치 크기 조절로 OOME 방지"""
    batch_size = initial_batch_size
    successful_count = 0
    failed_indices = []
    checkpoint_data = []

    total_docs = len(documents)
    i = 0

    while i < total_docs:
        batch_end = min(i + batch_size, total_docs)
        batch_docs = documents[i:batch_end]

        try:
            # 메모리 사전 정리
            import gc

            gc.collect()

            # 문서 크기 체크 및 제한
            batch_docs_filtered = []
            for doc in batch_docs:
                if len(doc.context) > 10000:  # 10KB 텍스트 제한
                    logger.warning(
                        f"Document too large ({len(doc.context)} chars), truncating"
                    )
                    doc.context = doc.context[:10000]
                batch_docs_filtered.append(doc)

            sentence_pairs = [doc.context for doc in batch_docs_filtered]

            # 극도로 메모리 효율적인 임베딩 생성
            dense_embedding = embedding_model.encode(
                sentence_pairs,
                normalize_embeddings=True,
                batch_size=1,  # 무조건 1개씩 임베딩
                show_progress_bar=False,
            )

            # Sparse 벡터 생성
            sparse_vectors = []
            for sentence in sentence_pairs:
                try:
                    sparse_vectors.append(create_sparse_vector(sentence))
                except Exception as e:
                    logger.error(f"Sparse vector creation failed: {e}")
                    sparse_vectors.append(None)

            # 포인트 생성 (메모리 효율적으로)
            points = []
            for index, doc in enumerate(batch_docs_filtered):
                if sparse_vectors[index] is not None:
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
                        {
                            "doc_index": i + index,
                            "point_id": point_id,
                            "doc_id": doc.ids,
                        }
                    )

                    # 메모리 절약을 위해 중간에 가비지 컬렉션
                    if len(points) % 10 == 0:
                        gc.collect()

            # Upsert with retry
            success, result = await upsert_batch_with_retry(collection_name, points)

            if success:
                successful_count += len(points)
                logger.info(f"Batch {i}-{batch_end} succeeded ({len(points)} items)")

                # 성공 시 배치 크기 유지 (최대 4 한계)
                if batch_size < 101:  # 최대 4까지만
                    batch_size = min(batch_size + 1, 100)
                    logger.debug(f"Increasing batch size to {batch_size}")

                i = batch_end
            else:
                # 실패 시 배치 크기 감소
                if batch_size > min_batch_size:
                    batch_size = max(batch_size // 2, min_batch_size)
                    logger.warning(
                        f"Reducing batch size to {batch_size} due to failure"
                    )
                else:
                    # 최소 배치 크기에서도 실패하면 건너뛰고 기록
                    failed_indices.extend(range(i, batch_end))
                    logger.error(
                        f"Skipping batch {i}-{batch_end} after repeated failures"
                    )
                    i = batch_end

        except MemoryError:
            # OOME 발생 시 즉시 1로 감소
            logger.error(f"CRITICAL: Memory error at batch size {batch_size}")
            batch_size = 1
            logger.warning(f"Emergency: reducing batch size to 1 due to memory error")

            # 강력한 메모리 정리
            import gc

            gc.collect()

            # 약간의 대기 시간 추가
            await asyncio.sleep(2)

            if batch_size == min_batch_size:
                # 1개에서도 메모리 에러면 문서가 너무 큰 것
                logger.critical(f"Cannot process document at index {i} - too large")
                failed_indices.extend(range(i, batch_end))
                i = batch_end

        except Exception as e:
            logger.error(f"Unexpected error in batch {i}-{batch_end}: {str(e)}")
            failed_indices.extend(range(i, batch_end))
            i = batch_end

    return {
        "successful_count": successful_count,
        "failed_count": len(failed_indices),
        "failed_indices": failed_indices,
        "checkpoint_data": checkpoint_data,
    }


async def verify_upsert_integrity(
    collection_name: str,
    checkpoint_data: List[Dict],
    sample_rate: float = 0.1,
) -> Dict[str, Any]:
    """업서트 후 무결성 검증"""
    if not checkpoint_data:
        return {"verified": True, "message": "No data to verify"}

    # 샘플링 검증 (전체 검증은 부하가 클 수 있음)
    sample_size = max(1, int(len(checkpoint_data) * sample_rate))
    sample_indices = random.sample(range(len(checkpoint_data)), sample_size)

    verified_count = 0
    missing_ids = []

    for idx in sample_indices:
        checkpoint = checkpoint_data[idx]
        point_id = checkpoint["point_id"]

        try:
            # 포인트 존재 확인
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
            logger.error(f"Verification failed for point {point_id}: {e}")
            missing_ids.append(point_id)

    integrity_rate = verified_count / sample_size if sample_size > 0 else 0

    return {
        "verified": integrity_rate >= 0.95,  # 95% 이상 검증 시 성공
        "integrity_rate": integrity_rate,
        "sample_size": sample_size,
        "missing_count": len(missing_ids),
        "missing_ids": missing_ids[:10],  # 최대 10개만 반환
    }


@app.post("/api/v1/mongo/page")
async def update_pages_to_mongo(pages=Body(...)):
    await pages_to_mongo(pages["pages"])
    return {"status": "ok"}


@app.post("/api/v2/documents", operation_id="upsert_documents_v2")
async def upsert_documents_v2(
    request: UpsertDocument,
):
    await create_collection(CreateCollection(collection_name=request.collection_name))
    count = 0
    total_document_length = len(request.documents)
    logger.info(f"upsert {total_document_length} items requested")
    for i in range(0, total_document_length, 100):
        try:
            batch_docs = request.documents[i : i + 100]
            sentence_pairs = [doc.context for doc in batch_docs]
            dense_embedding = embedding_model.encode(
                sentence_pairs, normalize_embeddings=True
            )
            sparse_vector = []
            for sentence in sentence_pairs:
                sparse_vector.append(create_sparse_vector(sentence))
            points = []
            for index, doc in enumerate(batch_docs):
                points.append(
                    models.PointStruct(
                        id=get_chunk_id(),
                        vector={
                            "dense": dense_embedding[index].tolist(),
                            "sparse": sparse_vector[index],
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
            result = await qdrant_client.upsert(
                collection_name=request.collection_name, points=points
            )
            count += len(points)
            logger.info(result.model_dump())
        except Exception as e:
            logger.error(str(e))
    return {"result": f"{count} items added."}


@app.post("/api/v1/documents", operation_id="upsert_documents")
async def upsert_documents(request: UpsertDocument):
    """개선된 문서 업서트 with 무결성 보장"""

    await create_collection(CreateCollection(collection_name=request.collection_name))

    total_document_length = len(request.documents)
    logger.info(f"Starting upsert of {total_document_length} items")

    # 적응형 배치 업서트 실행
    result = await adaptive_batch_upsert(
        collection_name=request.collection_name,
        documents=request.documents,
        initial_batch_size=100,  # 기존 크기부터 시작
        min_batch_size=5,  # 문제 시에만 1개씩
    )

    # 무결성 검증
    verification = await verify_upsert_integrity(
        collection_name=request.collection_name,
        checkpoint_data=result["checkpoint_data"],
        sample_rate=0.1,
    )

    # 실패한 문서 재시도 (옵션)
    retry_count = 0
    if result["failed_indices"] and len(result["failed_indices"]) < 50:
        logger.info(f"Retrying {len(result['failed_indices'])} failed documents")
        failed_docs = [request.documents[i] for i in result["failed_indices"]]
        retry_result = await adaptive_batch_upsert(
            collection_name=request.collection_name,
            documents=failed_docs,
            initial_batch_size=30,  # 재시도는 작게 시작
            min_batch_size=1,
        )
        retry_count = retry_result["successful_count"]

    final_success_count = result["successful_count"] + retry_count
    final_failed_count = result["failed_count"] - retry_count

    response = {
        "result": f"{final_success_count}/{total_document_length} items added successfully",
        "details": {
            "total_requested": total_document_length,
            "successful": final_success_count,
            "failed": final_failed_count,
            "integrity_verified": verification["verified"],
            "integrity_rate": verification.get("integrity_rate", 0),
        },
    }

    if final_failed_count > 0:
        logger.warning(f"Failed to upsert {final_failed_count} documents")
        response["warning"] = f"{final_failed_count} documents failed to upsert"

    return response


def create_sparse_vector(text):
    embeddings = list(sparse_embedding_model.embed([text]))[0]
    if hasattr(embeddings, "indices") and hasattr(embeddings, "values"):
        sparse_vector = models.SparseVector(
            indices=embeddings.indices.tolist(), values=embeddings.values.tolist()
        )
        return sparse_vector
    else:
        raise ValueError(
            "The embeddings object does not have 'indices' and 'values' attributes."
        )


@app.delete(
    "/api/v1/documents/{collection_name}/{id}",
    operation_id="remove_document",
    description="collection_name : html encoded string",
)
async def remove_document(
    collection_name: str,
    id: str,
):
    decoded_collection_name = html.unescape(collection_name)
    result = await qdrant_client.delete(
        decoded_collection_name,
        points_selector=models.Filter(
            must=[models.FieldCondition(key="id", match=models.MatchValue(value=id))]
        ),
    )
    return {"result": result.model_dump()}


@app.post("/api/v1/query/normalize")
async def query_normalize(query: str = Body(...)):
    normailzed = await query_normalizer(query)
    return normailzed


async def get_page(chunks):
    create_documents_index()
    mongo_client = AsyncIOMotorClient(
        f"mongodb://{settings.db.host}:{settings.db.port}"
    )

    page_collection = mongo_client["rag-data"]["documents"]

    try:
        p = []
        # 💡 개선점 1: 중복 p_id를 O(1) 시간 복잡도로 체크하기 위한 set
        added_p_ids = set()
        if isinstance(chunks[0], tuple):
            for chunk, score in chunks:
                # 'faq' 타입이 아닌 경우에만 DB에서 paragraph를 조회
                if chunk.get("metadatas", {}).get("paragraph_type") != "faq":
                    doc_id = chunk.get("metadatas", {}).get("doc_id")
                    paragraph_id = chunk.get("metadatas", {}).get("paragraph_id")
                    collection_name = chunk.get("metadatas", {}).get("collection_id")
                    # id 정보가 없는 chunk는 건너뜀
                    if not doc_id or not paragraph_id:
                        p.append(chunk)
                        continue

                    page = await page_collection.find_one(
                        {
                            "collection_id": collection_name,
                            "metadatas.doc_id": doc_id,
                            "paragraph_id": paragraph_id,
                        }
                    )

                    if page:
                        p_id = page.get("paragraph_id")
                        page["id"] = str(page["_id"])
                        page["rerank_score"] = float(score)
                        del page["_id"]
                        page["metadatas"]["bbox"] = chunk["metadatas"]["bbox"]
                        if p_id and p_id not in added_p_ids:
                            p.append(page)
                            added_p_ids.add(p_id)  # set에 p_id 추가
                    else:
                        # DB에서 파라그래프를 찾지 못한 경우 원본 chunk를 추가
                        p.append(chunk)
                else:
                    # 'faq' 타입인 경우 원본 chunk를 그대로 추가
                    p.append(chunk)
        else:
            for chunk in chunks:
                # 'faq' 타입이 아닌 경우에만 DB에서 paragraph를 조회
                if chunk.get("metadatas", {}).get("paragraph_type") != "faq":
                    doc_id = chunk.get("metadatas", {}).get("doc_id")
                    paragraph_id = chunk.get("metadatas", {}).get("paragraph_id")
                    collection_name = chunk.get("metadatas", {}).get("collection_id")
                    # id 정보가 없는 chunk는 건너뜀
                    if not doc_id or not paragraph_id:
                        p.append(chunk)
                        continue

                    page = await page_collection.find_one(
                        {
                            "collection_id": collection_name,
                            "metadatas.doc_id": doc_id,
                            "paragraph_id": paragraph_id,
                        }
                    )

                    if page:
                        p_id = page.get("paragraph_id")
                        page["id"] = str(page["_id"])
                        del page["_id"]
                        if "bbox" in page["metadatas"]:
                            page["metadatas"]["bbox"] = chunk["metadatas"]["bbox"]
                        if p_id and p_id not in added_p_ids:
                            p.append(page)
                            added_p_ids.add(p_id)  # set에 p_id 추가
                    else:
                        # DB에서 파라그래프를 찾지 못한 경우 원본 chunk를 추가
                        p.append(chunk)
                else:
                    # 'faq' 타입인 경우 원본 chunk를 그대로 추가
                    p.append(chunk)
        # 💡 개선점 4: 클라이언트 연결은 함수 외부에서 관리하므로 여기서 close() 호출 제거
        return p
    finally:
        mongo_client.close()


async def get_paragraphs(chunks):
    mongo_client = AsyncIOMotorClient(
        f"mongodb://{settings.db.host}:{settings.db.port}"
    )
    try:
        p = []
        # 💡 개선점 1: 중복 p_id를 O(1) 시간 복잡도로 체크하기 위한 set
        added_p_ids = set()

        for chunk in chunks:
            # 'faq' 타입이 아닌 경우에만 DB에서 paragraph를 조회
            if chunk.get("metadatas", {}).get("paragraph_type") != "faq":
                doc_id = chunk.get("metadatas", {}).get("doc_id")
                paragraph_id = chunk.get("metadatas", {}).get("paragraph_id")

                # id 정보가 없는 chunk는 건너뜀
                if not doc_id or not paragraph_id:
                    p.append(chunk)
                    continue

                parag = await get_paragraph(doc_id, paragraph_id, mongo_client)

                if parag:
                    p_id = parag.get("p_id")
                    # 💡 개선점 1, 2, 3 통합:
                    # 1. p_id가 있고,
                    # 2. added_p_ids set에 아직 추가되지 않았다면
                    if p_id and p_id not in added_p_ids:
                        p.append(parag)
                        added_p_ids.add(p_id)  # set에 p_id 추가
                else:
                    # DB에서 파라그래프를 찾지 못한 경우 원본 chunk를 추가
                    p.append(chunk)
            else:
                # 'faq' 타입인 경우 원본 chunk를 그대로 추가
                p.append(chunk)

        # 💡 개선점 4: 클라이언트 연결은 함수 외부에서 관리하므로 여기서 close() 호출 제거
        return p
    finally:
        mongo_client.close()


async def get_paragraph(doc_id, paragraph_id, mongo_client: AsyncIOMotorClient):
    db = mongo_client["paragraph"][doc_id]
    res = await db.find_one({"p_id": paragraph_id})
    if res and "_id" in res:
        del res["_id"]

    if res and "sentenses" in res:
        joined_context = []
        for p in res["sentenses"]:
            context = p["context"]
            if context.startswith(p["metadatas"]["prefix"]) == False:
                context = f"[{p["metadatas"]["prefix"]}] {context}"

            joined_context.append(context)

        paragraph = {
            "p_id": res["p_id"],
            "context": "\n".join(joined_context),
            "metadatas": {
                "doc_id": res["metadatas"]["doc_id"],
                "file_name": res["metadatas"]["file_name"],
            },
        }
    else:
        paragraph = res
    return paragraph


class SearchDocument(BaseModel):
    collection_name: str
    query: str
    use_paragraph: bool = False
    query_normalize: bool = False
    metadata_filter_key: str | None
    match_values: List[str] | None
    top_k: int = 100


@app.post("/api/v1/document/paragraph", operation_id="search_and_paragraph")
async def search_and_paragraph(request: SearchDocument):
    query = request.query
    if request.query_normalize:
        query = await query_normalizer(request.query)
    dense_query = embedding_model.encode(query, normalize_embeddings=True)
    sparse_query = list(sparse_embedding_model.embed([query]))[0]
    query_filter = None
    prefetch = [
        models.Prefetch(
            query=models.SparseVector(
                indices=sparse_query.indices.tolist(),
                values=sparse_query.values.tolist(),
            ),
            using="sparse",
            limit=request.top_k,
            score_threshold=0.6,
        ),
        models.Prefetch(
            query=dense_query, using="dense", limit=request.top_k, score_threshold=0.6
        ),
    ]
    if request.metadata_filter_key and len(request.match_values) > 0:
        filter_key = f"metadatas.{request.metadata_filter_key}"
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key=filter_key, match=models.MatchValue(value=request.match_values)
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
    )
    reranked = grid_rerank(query, fusion_results.points, 5)
    if request.use_paragraph:
        p = await get_paragraphs(reranked)  # 💡 await 추가
        return JSONResponse(content=p)
    else:
        return JSONResponse(content=reranked)


@app.post(
    "/api/v1/document/search_rerank",
    operation_id="search_similar_documents_with_reranker",
)
async def search_similar_documents_with_reranker(request: SearchDocument):
    start_time = time.time()
    try:
        query = request.query
        if request.query_normalize:
            query = await query_normalizer(request.query)
        dense_query = embedding_model.encode(query, normalize_embeddings=True)
        sparse_query = list(sparse_embedding_model.embed([query]))[0]
        query_filter = None
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
        if request.metadata_filter_key and len(request.match_values) > 0:
            filter_key = f"metadatas.{request.metadata_filter_key}"

            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key=filter_key, match=models.MatchAny(any=request.match_values)
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
            score_threshold=0.1,
        )
        reranked = grid_rerank(query, fusion_results.points, 5)
        if request.use_paragraph:
            p = await get_paragraphs(reranked)
            logger.info(f"{query} :\n{json.dumps(p, ensure_ascii=False, indent=True)}")
            return JSONResponse(content=p)
        else:
            return JSONResponse(content=reranked)
    except Exception as e:
        return JSONResponse(status_code=500, content=str(e))
    finally:
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"Search time: {execution_time:.5f} sec")


@app.post(
    "/api/v2/document/search_rerank",
    operation_id="search_similar_documents_with_reranker_v2",
)
async def search_similar_documents_with_reranker_v2(request: SearchDocument):
    start_time = time.time()
    try:
        query = request.query
        if request.query_normalize:
            query = await query_normalizer(request.query)
        dense_query = embedding_model.encode(query, normalize_embeddings=True)
        sparse_query = list(sparse_embedding_model.embed([query]))[0]
        query_filter = None
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
        if request.metadata_filter_key and len(request.match_values) > 0:
            filter_key = f"metadatas.{request.metadata_filter_key}"

            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key=filter_key, match=models.MatchAny(any=request.match_values)
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
            score_threshold=0.1,
        )
        reranked = grid_rerank(query, fusion_results.points, 5)
        if request.use_paragraph:
            p = await get_page(reranked)
            logger.info(f"{query} :\n{json.dumps(p, ensure_ascii=False, indent=True)}")
            return JSONResponse(content=p)
        else:
            return JSONResponse(content=reranked)
    except Exception as e:
        return JSONResponse(status_code=500, content=str(e))
    finally:
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"Search time: {execution_time:.5f} sec")


@app.post(
    "/api/v1/document/search_test",
    operation_id="search_similar_documents_test_with_reranker",
)
async def search_similar_documents_test_with_reranker(request: SearchDocument):
    query = request.query
    if request.query_normalize:
        query = await query_normalizer(request.query)
    dense_query = embedding_model.encode(query, normalize_embeddings=True)
    sparse_query = list(sparse_embedding_model.embed([query]))[0]
    query_filter = None
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
    if request.metadata_filter_key and len(request.match_values) > 0:
        filter_key = f"metadatas.{request.metadata_filter_key}"
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key=filter_key, match=models.MatchAny(any=request.match_values)
                )
            ]
        )
    fusion_results = await qdrant_client.query_points(
        collection_name=request.collection_name,
        prefetch=prefetch,
        query_filter=query_filter,
        search_params=models.SearchParams(hnsw_ef=128),
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=50,
    )
    response = {
        "fusion_result": [
            {
                "context": result.payload["context"],
                "ids": result.payload["ids"],
                "metadatas": result.payload["metadatas"],
            }
            for result in fusion_results.points
        ]
    }
    reranked = grid_rerank(query, fusion_results.points, 5)
    response["reranked"] = reranked
    response["query"] = query
    p = await get_paragraphs(reranked)
    logger.info(f"{query} :\n{json.dumps(p, ensure_ascii=False, indent=True)}")
    response["paragraph"] = p
    return JSONResponse(content=response)


@app.post("/api/v1/document/search", operation_id="search_document")
async def search_similar_documents(request: SearchDocument):
    query = request.query
    if request.query_normalize:
        query = await query_normalizer(request.query)
    dense_query = embedding_model.encode(query, normalize_embeddings=True)
    sparse_query = list(sparse_embedding_model.embed([query]))[0]
    query_filter = None
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
    if request.metadata_filter_key and len(request.match_values) > 0:
        filter_key = f"metadatas.{request.metadata_filter_key}"
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key=filter_key, match=models.MatchValue(value=request.match_values)
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


async def pages_to_mongo(paragraph_list):
    create_documents_index()
    mongo_client = AsyncIOMotorClient(
        f"mongodb://{settings.db.host}:{settings.db.port}"
    )
    doc_collection = mongo_client["rag-data"]["documents"]
    await doc_collection.insert_many(paragraph_list)


async def paragraph_to_mongo(document_id, paragraph_list):
    """MongoDB에 문단 저장 with retry"""
    if not paragraph_list:
        return

    mongo_client = AsyncIOMotorClient(
        f"mongodb://{settings.db.host}:{settings.db.port}"
    )
    try:
        doc_collection = mongo_client["paragraph"][document_id]

        # 배치 처리로 메모리 효율성 향상
        batch_size = 100
        for i in range(0, len(paragraph_list), batch_size):
            batch = paragraph_list[i : i + batch_size]
            retry_count = 3
            while retry_count > 0:
                try:
                    await doc_collection.insert_many(batch, ordered=False)
                    break
                except Exception as e:
                    retry_count -= 1
                    if retry_count == 0:
                        logger.error(
                            f"Failed to insert paragraph batch {i}-{i+batch_size}: {e}"
                        )
                    else:
                        await asyncio.sleep(2 ** (3 - retry_count))
    finally:
        mongo_client.close()


async def chunk_to_mongo(collection_name, chunks):
    """MongoDB에 청크 저장 with retry"""
    if not chunks:
        return

    mongo_client = AsyncIOMotorClient(
        f"mongodb://{settings.db.host}:{settings.db.port}"
    )
    try:
        doc_collection = mongo_client["chunks"][collection_name]

        # 배치 처리로 메모리 효율성 향상
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
                        logger.error(
                            f"Failed to insert batch {i}-{i+batch_size} to MongoDB: {e}"
                        )
                    else:
                        await asyncio.sleep(2 ** (3 - retry_count))
    finally:
        mongo_client.close()


# @app.post("/api/v2/document/search_page")
# async def search_similar_documents_with_reranker_v2(
#     request: SearchDocument
# ):
#     start_time = time.time()
#     try:
#         query = request.query
#         if request.query_normalize:
#             query = await query_normalizer(request.query)
#         dense_query = embedding_model.encode(query, normalize_embeddings=True)
#         sparse_query = list(sparse_embedding_model.embed([query]))[0]
#         query_filter = None
#         prefetch = [
#             models.Prefetch(
#                 query=models.SparseVector(
#                     indices=sparse_query.indices.tolist(),
#                     values=sparse_query.values.tolist(),
#                 ),
#                 using="sparse",
#                 limit=request.top_k,
#             ),
#             models.Prefetch(query=dense_query, using="dense", limit=request.top_k),
#         ]
#         if request.metadata_filter_key and len(request.match_values) > 0:
#             filter_key = f"metadatas.{request.metadata_filter_key}"

#             query_filter = models.Filter(
#                 must=[
#                     models.FieldCondition(
#                         key=filter_key, match=models.MatchAny(any=request.match_values)
#                     )
#                 ]
#             )

#         fusion_results = await qdrant_client.query_points(
#             collection_name=request.collection_name,
#             prefetch=prefetch,
#             query_filter=query_filter,
#             search_params=models.SearchParams(hnsw_ef=128),
#             query=models.FusionQuery(fusion=models.Fusion.RRF),
#             limit=request.top_k,
#             score_threshold=0.1,
#         )

#         reranked = qwen3_rerank(query, fusion_results.points)
#         p = await get_page(reranked)
#         logger.info(f"{query} :\n{json.dumps(p, ensure_ascii=False, indent=True)}")
#         return JSONResponse(content=p)

#     except Exception as e:
#         return JSONResponse(status_code=500, content=str(e))
#     finally:
#         end_time = time.time()
#         execution_time = end_time - start_time
#         logger.info(f"Search time: {execution_time:.5f} sec")


@app.post(
    "/api/v1/document/search_page",
    operation_id="search_page",
)
async def search_page(request: SearchDocument):
    start_time = time.time()
    try:
        query = request.query
        if request.query_normalize:
            query = await query_normalizer(request.query)
        dense_query = embedding_model.encode(query, normalize_embeddings=True)
        sparse_query = list(sparse_embedding_model.embed([query]))[0]
        query_filter = None
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
        if request.metadata_filter_key and len(request.match_values) > 0:
            filter_key = f"metadatas.{request.metadata_filter_key}"

            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key=filter_key, match=models.MatchAny(any=request.match_values)
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
            score_threshold=0.1,
        )
        reranked = grid_rerank(query, fusion_results.points, 5)
        p = await get_page(reranked)
        logger.info(f"{query} :\n{json.dumps(p, ensure_ascii=False, indent=True)}")
        return JSONResponse(content=p)

    except Exception as e:
        return JSONResponse(status_code=500, content=str(e))
    finally:
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"Search time: {execution_time:.5f} sec")


# @torch.no_grad()
# def compute_scores(query, points: List[str], batch_size=16) -> np.ndarray:
#     scores = []

#     # 배치 처리
#     for i in range(0, len(points), batch_size):
#         batch_docs = points[i : i + batch_size]

#         # 쿼리-문서 쌍 생성
#         pairs = [[query, doc] for doc in batch_docs]

#         # 토크나이징
#         inputs = tokenizer(
#             pairs,
#             padding=True,
#             truncation=True,
#             max_length=8196,
#             return_tensors="pt",
#         ).to(device)

#         # 점수 계산
#         outputs = model(**inputs)

#         # logits를 점수로 변환 (시그모이드 또는 소프트맥스 적용 가능)
#         if outputs.logits.shape[-1] == 1:
#             # 회귀 모델인 경우
#             batch_scores = outputs.logits.squeeze(-1).cpu().numpy()
#         else:
#             # 분류 모델인 경우 (positive 클래스의 확률 사용)
#             batch_scores = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()

#         scores.extend(batch_scores)
#     return np.array(scores)


# def qwen3_rerank(
#     query: str,
#     documents: List[models.ScoredPoint],
#     top_k: Optional[int] = 5,
#     return_scores: bool = False,
# ) -> List[Tuple[Document, float]]:
#     """
#     문서들을 리랭킹

#     Args:
#         query: 검색 쿼리
#         documents: Document 객체 리스트
#         top_k: 반환할 상위 문서 개수 (None이면 전체 반환)
#         return_scores: 점수를 함께 반환할지 여부

#     Returns:
#         (Document, score) 튜플 리스트 (점수 내림차순 정렬)
#     """
#     if not documents:
#         return []

#     # 문서 텍스트 추출
#     doc_texts = [doc.payload["context"] for doc in documents]

#     # 점수 계산
#     logger.info(f"Computing scores for {len(documents)} documents...")
#     scores = compute_scores(query, doc_texts)

#     # 점수와 문서를 쌍으로 묶기
#     doc_score_pairs = list(zip(documents, scores))

#     # 점수 기준 내림차순 정렬
#     doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

#     # top_k 적용
#     if top_k is not None:
#         doc_score_pairs = doc_score_pairs[:top_k]

#     if return_scores:
#         return doc_score_pairs
#     else:
#         return [(doc.payload, score) for doc, score in doc_score_pairs]


def grid_rerank(query, fusion, rerank_top_k=5):
    url = f"{settings.a2o.reranker}/v2/predict"
    chunks = [
        {
            "context": result.payload["context"],
            "ids": result.payload["ids"],
            "metadatas": result.payload["metadatas"],
        }
        for result in fusion
    ]
    packet = {
        "reqid": "1818",
        "input": {
            "items": [{"query": query, "k": rerank_top_k, "retrieved_chunks": chunks}]
        },
    }
    payload = json.dumps(packet, ensure_ascii=False)
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.request(
            "POST", url, headers=headers, data=payload, timeout=10000
        )
        res_json = json.loads(response.text)
        col = res_json["output"]["items"][0]["top_k_chunks"]
        return col
    except:
        return []


class EmbeddingConsumer(RagEmbeddingWorker):
    async def handle_message(self, data: Dict[str, Any], context: MessageContext):
        mq = RabbitMQClient.get_instance()
        redis_client = Redis()

        try:
            # {
            #     "job_id": "unique-job-id",
            #     "file_id": "file-storage-id",
            #     "doc_id": "document-id",
            #     "collection_id": "default",
            #     "redis_chunk_key": "rag.document.{doc_id}.chunk",
            #     "redis_para_key": "rag.document.{doc_id}.para",
            #     "redis_img_key": "rag.document.{doc_id}.img",
            #     "redis_tbl_key": "rag.document.{doc_id}.tbl",
            # }

            job_id = data.get("job_id")

            logger.info(f"[{job_id}] Embedding Reqeust Incomming...")
            await mq.get_producer(RagProducer.EMBEDDING_STARTED).publish(
                {"job_id": job_id, "message": "임베딩 시작"}
            )
            logger.info(f"[{job_id}] Start embedding...")

            doc_id = data.get("doc_id")
            file_id = data.get("file_id")
            collection_id = data.get("collection_id")
            redis_para_key = data.get("redis_para_key")
            redis_chunk_key = data.get("redis_chunk_key")
            redis_img_key = data.get("redis_img_key", None)
            redis_tbl_key = data.get("redis_tbl_key", None)

            logger.info(json.dumps(data, ensure_ascii=False, indent=True))

            chunks = await redis_client.get(redis_chunk_key)
            paragraphs = await redis_client.get(redis_para_key)

            # 디버깅: Redis에서 받은 데이터 타입 확인
            logger.debug(
                f"chunks type: {type(chunks)}, value: {chunks if chunks is None else 'Has data'}"
            )
            logger.debug(
                f"paragraphs type: {type(paragraphs)}, value: {paragraphs if paragraphs is None else 'Has data'}"
            )

            # 안전한 데이터 추출 (dict 체크 + get 메서드 사용)
            c_data = []
            if chunks and isinstance(chunks, dict):
                c_data = chunks.get("data", [])
            c_data_len = len(c_data)
            logger.info(f"chunks : {c_data_len} items")

            p_data = []
            if paragraphs and isinstance(paragraphs, dict):
                p_data = paragraphs.get("data", [])
            p_data_len = len(p_data)
            logger.info(f"paragraphs : {p_data_len} items")

            request = UpsertDocument(collection_name=collection_id, documents=c_data)
            result = await upsert_documents(request)
            logger.info(json.dumps(result, ensure_ascii=False, indent=True))
            await chunk_to_mongo(collection_id, c_data)
            logger.info(f"{c_data_len} chunks added.(mongo)")
            await pages_to_mongo(p_data)
            logger.info(f"{p_data_len} paragraphs added.(mongo)")

            await mq.get_producer(RagProducer.EMBEDDING_COMPLETED).publish(
                {
                    "job_id": job_id,
                    "file_id": file_id,
                    "embedded_chunks": c_data_len,
                    "embedded_paragraphs": p_data_len,
                    "redis_para_key": redis_para_key,
                    "redis_chunk_key": redis_chunk_key,
                    "redis_img_key": redis_img_key,
                    "redis_tbl_key": redis_tbl_key,
                    "message": "임베딩 완료",
                }
            )
            logger.info(f"[{job_id}] Embedding completed successfully")

        except Exception as e:
            failure_message = {
                "job_id": job_id,
                "error": str(e),
                "error_type": type(e).__name__,
                "handler": handler_name,
            }

            message = json.dumps(failure_message, ensure_ascii=False, indent=True)
            await mq.get_producer(RagProducer.EMBEDDING_FAILED).publish(
                {
                    "job_id": job_id,
                    "doc_id": doc_id,
                    "collection_id": collection_id,
                    "redis_para_key": redis_para_key,
                    "redis_chunk_key": redis_chunk_key,
                    "redis_img_key": redis_img_key,
                    "redis_tbl_key": redis_tbl_key,
                    "message": message,
                }
            )
            logger.error(f"[{job_id}] {message}")


rabbitmq_connection_config = ConnectionConfig(**settings.mq.model_dump())


async def setup_redis():
    redis_client = Redis()
    await redis_client.init(**settings.redis.model_dump())


async def clear_redis():
    redis_client = Redis()
    await redis_client.close()


async def setup_mq():
    rabbitmq_client = RabbitMQClient.get_instance()
    await rabbitmq_client.init(rabbitmq_connection_config)

    from gridone_rabbitmq.rag import (
        publish_embedding_started,
        publish_embedding_completed,
        publish_embedding_failed,
    )

    await rabbitmq_client.register_handlers(
        [
            publish_embedding_started,
            publish_embedding_completed,
            publish_embedding_failed,
        ]
    )
    rabbitmq_client.add_consumer(EmbeddingConsumer(rabbitmq_client))
    # await rabbitmq_client.start_consumers()
    return rabbitmq_client


async def clear_mq():
    rabbitmq_client = RabbitMQClient.get_instance()
    await rabbitmq_client.close()
