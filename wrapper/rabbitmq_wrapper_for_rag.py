"""
wrapper_rabbitmq_for_rag
====================
RAG 파이프라인 전용 Producer / Consumer 정의.

공개 심볼
---------
RagProducer              : Producer 식별 Enum
RagEmbeddingWorker       : 임베딩 Consumer 기반 클래스
publish_embedding_started   : ProducerConfig (register_handlers에 전달)
publish_embedding_completed : ProducerConfig
publish_embedding_failed    : ProducerConfig

큐 이름 규칙
-----------
소비 (Consumer)
  rag.embedding.request     임베딩 작업 요청 수신

발행 (Producer)
  rag.embedding.started     작업 시작 알림
  rag.embedding.completed   작업 완료 알림
  rag.embedding.failed      작업 실패 알림
"""

from __future__ import annotations

from abc import abstractmethod
from enum import Enum
from typing import Any, Dict

from wrapper.rabbitmq_wrapper import (
    BaseConsumer,
    MessageContext,
    ProducerConfig,
    RabbitMQClient,
)


# ---------------------------------------------------------------------------
# Producer 식별자 Enum
# ---------------------------------------------------------------------------


class RagProducer(str, Enum):
    EMBEDDING_STARTED = "rag.embedding.started"
    EMBEDDING_COMPLETED = "rag.embedding.completed"
    EMBEDDING_FAILED = "rag.embedding.failed"


# ---------------------------------------------------------------------------
# ProducerConfig 인스턴스 (register_handlers에 넘길 객체들)
# ---------------------------------------------------------------------------

publish_embedding_started = ProducerConfig(
    producer_type=RagProducer.EMBEDDING_STARTED,
    queue=RagProducer.EMBEDDING_STARTED.value,
)

publish_embedding_completed = ProducerConfig(
    producer_type=RagProducer.EMBEDDING_COMPLETED,
    queue=RagProducer.EMBEDDING_COMPLETED.value,
)

publish_embedding_failed = ProducerConfig(
    producer_type=RagProducer.EMBEDDING_FAILED,
    queue=RagProducer.EMBEDDING_FAILED.value,
)


# ---------------------------------------------------------------------------
# 임베딩 Consumer 기반 클래스
# ---------------------------------------------------------------------------


class RagEmbeddingWorker(BaseConsumer):
    """
    임베딩 작업 요청을 수신하는 Consumer 기반 클래스.

    서브클래스에서 handle_message()를 구현합니다::

        class EmbeddingConsumer(RagEmbeddingWorker):
            async def handle_message(self, data, context):
                job_id = data.get("job_id")
                ...
    """

    queue: str = "rag.embedding.request"
    durable: bool = True
    prefetch_count: int = 1

    def __init__(self, client: RabbitMQClient) -> None:
        super().__init__(client)

    @abstractmethod
    async def handle_message(
        self, data: Dict[str, Any], context: MessageContext
    ) -> None: ...
