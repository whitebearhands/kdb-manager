"""
kdb_manager.py
==============
RAG Knowledge DB Manager — 애플리케이션 진입점.

이 파일의 역할
--------------
- FastAPI 앱 생성 및 미들웨어 설정
- 앱 라이프사이클 (Redis · RabbitMQ 연결/해제)
- RabbitMQ EmbeddingConsumer 정의
  (MQ 메시지를 받아 임베딩 → Qdrant upsert → MongoDB 저장 수행)
- 라우터 등록

각 기능의 세부 구현은 routes/ 패키지에 분리되어 있다.
  routes/collection.py  — 컬렉션 CRUD, 문서 조회
  routes/document.py    — 문서 업서트, 삭제, MongoDB 저장
  routes/search.py      — 하이브리드 검색, 리랭킹, 단락 복원
  routes/feedback.py    — 검색 피드백
  routes/query_cache.py — 쿼리 캐싱
"""

import argparse
import contextlib
import json
from logging import getLogger
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from wrapper.logger_wrapper import setup_logger
from wrapper.rabbitmq_wrapper import ConnectionConfig, MessageContext, RabbitMQClient
from wrapper.rabbitmq_wrapper_for_rag import RagEmbeddingWorker, RagProducer
from wrapper.redis_wrapper import Redis

# 라우터
from routes.collection import router as CollectionRouter
from routes.document import (
    UpsertDocument,
    chunk_to_mongo,
    paragraph_to_mongo,
    router as DocumentRouter,
    upsert_documents,
)
from routes.feedback import router as FeedBackRouter
from routes.query_cache import router as QueryRouter
from routes.search import router as SearchRouter

# ── 로거 초기화 (가장 먼저 실행) ────────────────────────────────────────────
setup_logger(settings.log)
logger = getLogger(__name__)

# ── CLI 인수: --port로 포트 오버라이드 가능 ──────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, help="서비스 포트 번호 (기본값: settings.app.port)")
args, _ = parser.parse_known_args()
if args.port:
    settings.app.port = args.port

# ── EmbeddingConsumer가 실패 메시지에 포함할 서비스 식별자 ───────────────────
HANDLER_NAME = "kdb-manager"


# ════════════════════════════════════════════════════════════════════════════════
# 앱 라이프사이클
# ════════════════════════════════════════════════════════════════════════════════


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 앱 시작/종료 시 실행되는 라이프사이클 핸들러.

    시작 시
    -------
    1. Redis 연결 초기화
    2. RabbitMQ 연결 및 EmbeddingConsumer 등록

    종료 시
    -------
    1. RabbitMQ 연결 해제
    2. Redis 연결 해제
    """
    logger.info("서비스 시작 중...")
    await _setup_redis()
    await _setup_mq()
    logger.info(f"서비스 시작 완료 (port={settings.app.port})")

    yield  # 앱 실행 구간

    logger.info("서비스 종료 중...")
    await _clear_mq()
    await _clear_redis()
    logger.info("서비스 종료 완료")


# ════════════════════════════════════════════════════════════════════════════════
# FastAPI 앱 생성
# ════════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="KDB Manager",
    description="RAG 파이프라인을 위한 벡터 DB 관리 서비스",
    version="2.1.1",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 라우터 등록 ──────────────────────────────────────────────────────────────
app.include_router(CollectionRouter)
app.include_router(DocumentRouter)
app.include_router(SearchRouter)
app.include_router(FeedBackRouter)
app.include_router(QueryRouter)


# ════════════════════════════════════════════════════════════════════════════════
# RabbitMQ EmbeddingConsumer
# ════════════════════════════════════════════════════════════════════════════════


class EmbeddingConsumer(RagEmbeddingWorker):
    """
    RabbitMQ 'rag.embedding.request' 큐를 구독하는 임베딩 소비자.

    수신 메시지 형식 (Redis 키 기반)
    ----------------------------------
    {
        "job_id":          "unique-job-id",
        "file_id":         "file-storage-id",
        "doc_id":          "document-id",
        "collection_id":   "default",
        "redis_chunk_key": "rag.document.{doc_id}.chunk",
        "redis_para_key":  "rag.document.{doc_id}.para",
        "redis_img_key":   "rag.document.{doc_id}.img",   (선택)
        "redis_tbl_key":   "rag.document.{doc_id}.tbl"    (선택)
    }

    처리 순서
    ---------
    1. Redis에서 청크/단락 데이터 로드
    2. upsert_documents()로 청크 임베딩 → Qdrant 저장
    3. chunk_to_mongo()로 청크 원문 → MongoDB 저장
    4. paragraph_to_mongo()로 단락 원문 → MongoDB 저장
    5. 완료/실패 메시지를 MQ 프로듀서로 발행
    """

    async def handle_message(self, data: Dict[str, Any], context: MessageContext):
        mq = RabbitMQClient.get_instance()
        redis_client = Redis()

        job_id = data.get("job_id")
        doc_id = data.get("doc_id")
        file_id = data.get("file_id")
        collection_id = data.get("collection_id")
        redis_chunk_key = data.get("redis_chunk_key")
        redis_para_key = data.get("redis_para_key")
        redis_img_key = data.get("redis_img_key")
        redis_tbl_key = data.get("redis_tbl_key")

        try:
            logger.info(f"[{job_id}] 임베딩 요청 수신")
            await mq.get_producer(RagProducer.EMBEDDING_STARTED).publish(
                {"job_id": job_id, "message": "임베딩 시작"}
            )

            # Redis에서 청크/단락 데이터 로드
            chunks_raw = await redis_client.get(redis_chunk_key)
            paragraphs_raw = await redis_client.get(redis_para_key)

            # Redis 값은 {"data": [...]} 형식의 dict
            c_data = chunks_raw.get("data", []) if isinstance(chunks_raw, dict) else []
            p_data = paragraphs_raw.get("data", []) if isinstance(paragraphs_raw, dict) else []

            logger.info(f"[{job_id}] 청크 {len(c_data)}건 / 단락 {len(p_data)}건 로드")

            # 청크 임베딩 → Qdrant upsert
            upsert_request = UpsertDocument(
                collection_name=collection_id, documents=c_data
            )
            result = await upsert_documents(upsert_request)
            logger.info(f"[{job_id}] Qdrant upsert 완료: {result.get('result')}")

            # 원문 MongoDB 저장
            await chunk_to_mongo(collection_id, c_data)
            logger.info(f"[{job_id}] 청크 MongoDB 저장 완료: {len(c_data)}건")

            await paragraph_to_mongo(p_data)
            logger.info(f"[{job_id}] 단락 MongoDB 저장 완료: {len(p_data)}건")

            # 완료 이벤트 발행
            await mq.get_producer(RagProducer.EMBEDDING_COMPLETED).publish(
                {
                    "job_id": job_id,
                    "file_id": file_id,
                    "embedded_chunks": len(c_data),
                    "embedded_paragraphs": len(p_data),
                    "redis_chunk_key": redis_chunk_key,
                    "redis_para_key": redis_para_key,
                    "redis_img_key": redis_img_key,
                    "redis_tbl_key": redis_tbl_key,
                    "message": "임베딩 완료",
                }
            )
            logger.info(f"[{job_id}] 임베딩 완료")

        except Exception as e:
            # 실패 이벤트 발행 후 예외를 재raise하지 않음
            # (MQ 소비자가 메시지를 ack하고 실패 큐로 라우팅)
            failure = {
                "job_id": job_id,
                "doc_id": doc_id,
                "collection_id": collection_id,
                "error": str(e),
                "error_type": type(e).__name__,
                "handler": HANDLER_NAME,
            }
            await mq.get_producer(RagProducer.EMBEDDING_FAILED).publish(
                {
                    **failure,
                    "redis_chunk_key": redis_chunk_key,
                    "redis_para_key": redis_para_key,
                    "redis_img_key": redis_img_key,
                    "redis_tbl_key": redis_tbl_key,
                    "message": json.dumps(failure, ensure_ascii=False),
                }
            )
            logger.error(f"[{job_id}] 임베딩 실패: {e}")


# ════════════════════════════════════════════════════════════════════════════════
# Redis / RabbitMQ 초기화 헬퍼
# ════════════════════════════════════════════════════════════════════════════════

_rabbitmq_config = ConnectionConfig(**settings.mq.model_dump())


async def _setup_redis() -> None:
    """Redis 연결을 초기화한다 (싱글톤 Redis 인스턴스 재사용)."""
    redis_client = Redis()
    await redis_client.init(**settings.redis.model_dump())
    logger.info("Redis 연결 완료")


async def _clear_redis() -> None:
    """Redis 연결을 닫는다."""
    redis_client = Redis()
    await redis_client.close()
    logger.info("Redis 연결 해제")


async def _setup_mq() -> None:
    """
    RabbitMQ에 연결하고 프로듀서/컨슈머를 등록한다.

    등록 프로듀서
    -------------
    - EMBEDDING_STARTED   : 임베딩 시작 알림
    - EMBEDDING_COMPLETED : 임베딩 완료 알림
    - EMBEDDING_FAILED    : 임베딩 실패 알림

    등록 컨슈머
    -----------
    - EmbeddingConsumer : rag.embedding.request 큐 구독
      (start_consumers()는 주석 처리 — 필요 시 활성화)
    """
    from wrapper.rabbitmq_wrapper_for_rag import (
        publish_embedding_completed,
        publish_embedding_failed,
        publish_embedding_started,
    )

    client = RabbitMQClient.get_instance()
    await client.init(_rabbitmq_config)
    await client.register_handlers(
        [publish_embedding_started, publish_embedding_completed, publish_embedding_failed]
    )
    client.add_consumer(EmbeddingConsumer(client))
    # await client.start_consumers()  # 활성화 시 MQ 소비 시작
    logger.info("RabbitMQ 연결 및 핸들러 등록 완료")


async def _clear_mq() -> None:
    """RabbitMQ 연결을 닫는다."""
    client = RabbitMQClient.get_instance()
    await client.close()
    logger.info("RabbitMQ 연결 해제")
