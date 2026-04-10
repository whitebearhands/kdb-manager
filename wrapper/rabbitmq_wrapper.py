"""
wrapper_rabbitmq
================
asyncio 기반 RabbitMQ 클라이언트 래퍼.

- ConnectionConfig  : 연결 설정 (pydantic BaseModel)
- MessageContext    : 소비한 메시지의 라우팅 정보
- RabbitMQClient    : 싱글톤 클라이언트
  - init(config)              연결
  - register_handlers(list)   ProducerConfig 목록 등록 → Publisher 생성
  - get_producer(enum_value)  등록된 Publisher 반환
  - add_consumer(worker)      Consumer 추가
  - start_consumers()         소비 시작
  - close()                   연결 해제

의존: aio-pika  (pip install aio-pika)
"""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type

try:
    import aio_pika
    from aio_pika import ExchangeType, IncomingMessage, Message, RobustConnection
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "wrapper_rabbitmq requires 'aio-pika'. " "Install it with: pip install aio-pika"
    ) from exc

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 공개 설정/데이터 모델
# ---------------------------------------------------------------------------


class ConnectionConfig(BaseModel):
    """RabbitMQ 연결 설정."""

    host: str = Field(default="localhost")
    port: int = Field(default=5672)
    username: str = Field(default="guest")
    password: str = Field(default="guest")
    virtual_host: str = Field(default="/")
    ssl: bool = Field(default=False)
    heartbeat: int = Field(default=600)

    # 사용하지 않는 필드가 있어도 오류 없이 무시
    model_config = {"extra": "ignore"}


@dataclass
class MessageContext:
    """소비된 메시지의 라우팅 정보."""

    exchange: str
    routing_key: str
    message_id: Optional[str] = None
    headers: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProducerConfig:
    """
    Producer 정의 — register_handlers()에 넘기는 단위 객체.

    Parameters
    ----------
    producer_type : Any
        get_producer()로 조회할 때 사용하는 식별자 (주로 Enum 값).
    queue : str
        메시지를 발행할 큐 이름.
    exchange : str
        사용할 Exchange 이름. 기본값 "" → default exchange (direct).
    exchange_type : ExchangeType
        Exchange 타입. default exchange 사용 시 무시.
    durable : bool
        큐·Exchange 내구성 여부.
    """

    producer_type: Any
    queue: str
    exchange: str = ""
    exchange_type: ExchangeType = ExchangeType.DIRECT
    durable: bool = True


# ---------------------------------------------------------------------------
# Publisher (내부)
# ---------------------------------------------------------------------------


class Publisher:
    """등록된 ProducerConfig 하나에 대응하는 발행자."""

    def __init__(
        self,
        channel: aio_pika.RobustChannel,
        config: ProducerConfig,
    ) -> None:
        self._channel = channel
        self._config = config
        self._exchange: Optional[aio_pika.Exchange] = None

    async def _ensure_exchange(self) -> aio_pika.Exchange:
        if self._exchange is None:
            if self._config.exchange:
                self._exchange = await self._channel.declare_exchange(
                    self._config.exchange,
                    self._config.exchange_type,
                    durable=self._config.durable,
                )
            else:
                # default exchange
                self._exchange = self._channel.default_exchange
        return self._exchange

    async def publish(self, data: Dict[str, Any]) -> None:
        """data 딕셔너리를 JSON으로 직렬화해 발행합니다."""
        exchange = await self._ensure_exchange()
        body = json.dumps(data, ensure_ascii=False).encode()
        message = Message(
            body=body,
            content_type="application/json",
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
        )
        routing_key = (
            self._config.queue if not self._config.exchange else self._config.queue
        )
        await exchange.publish(message, routing_key=routing_key)
        logger.debug(f"Published to '{routing_key}': {data}")


# ---------------------------------------------------------------------------
# Consumer 기반 클래스 (공개)
# ---------------------------------------------------------------------------


class BaseConsumer(ABC):
    """소비자 기반 클래스."""

    # 서브클래스에서 반드시 지정
    queue: str = ""
    durable: bool = True
    prefetch_count: int = 1

    def __init__(self, client: "RabbitMQClient") -> None:
        self._client = client

    @abstractmethod
    async def handle_message(
        self, data: Dict[str, Any], context: MessageContext
    ) -> None:
        """메시지 처리 구현부."""

    async def _on_message(self, message: IncomingMessage) -> None:
        async with message.process(requeue=True):
            try:
                data = json.loads(message.body.decode())
                context = MessageContext(
                    exchange=message.exchange or "",
                    routing_key=message.routing_key or "",
                    message_id=message.message_id,
                    headers=dict(message.headers or {}),
                )
                await self.handle_message(data, context)
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e} | body={message.body!r}")
            except Exception as e:
                logger.exception(f"handle_message 오류: {e}")
                raise


# ---------------------------------------------------------------------------
# RabbitMQClient (싱글톤)
# ---------------------------------------------------------------------------


class RabbitMQClient:
    """
    RabbitMQ 연결·발행·소비를 관리하는 싱글톤 클라이언트.

    사용 예::

        config = ConnectionConfig(host="localhost", username="guest", password="guest")
        client = RabbitMQClient.get_instance()
        await client.init(config)

        # Producer 등록
        await client.register_handlers([my_producer_config])

        # Consumer 등록 & 시작
        client.add_consumer(MyConsumer(client))
        await client.start_consumers()

        # 메시지 발행
        await client.get_producer(MyProducerEnum.SOME_EVENT).publish({"key": "value"})

        await client.close()
    """

    _instance: Optional["RabbitMQClient"] = None

    def __init__(self) -> None:
        self._connection: Optional[RobustConnection] = None
        self._channel: Optional[aio_pika.RobustChannel] = None
        self._producers: Dict[Any, Publisher] = {}
        self._consumers: List[BaseConsumer] = []

    @classmethod
    def get_instance(cls) -> "RabbitMQClient":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # 연결
    # ------------------------------------------------------------------

    async def init(self, config: ConnectionConfig) -> None:
        """RabbitMQ에 연결하고 채널을 엽니다."""
        url = (
            f"{'amqps' if config.ssl else 'amqp'}://"
            f"{config.username}:{config.password}"
            f"@{config.host}:{config.port}/{config.virtual_host.lstrip('/')}"
            f"?heartbeat={config.heartbeat}"
        )
        self._connection = await aio_pika.connect_robust(url)
        self._channel = await self._connection.channel()
        logger.info(f"RabbitMQ 연결 완료: {config.host}:{config.port}")

    async def close(self) -> None:
        """연결을 닫습니다."""
        if self._connection and not self._connection.is_closed:
            await self._connection.close()
            logger.info("RabbitMQ 연결 종료")
        self._connection = None
        self._channel = None
        self._producers.clear()

    # ------------------------------------------------------------------
    # Producer
    # ------------------------------------------------------------------

    async def register_handlers(self, configs: List[ProducerConfig]) -> None:
        """
        ProducerConfig 목록을 등록합니다.
        각 큐를 선언하고 Publisher 인스턴스를 생성합니다.
        """
        if self._channel is None:
            raise RuntimeError("init()을 먼저 호출하세요.")

        for cfg in configs:
            await self._channel.declare_queue(cfg.queue, durable=cfg.durable)
            publisher = Publisher(self._channel, cfg)
            self._producers[cfg.producer_type] = publisher
            logger.debug(f"Producer 등록: {cfg.producer_type} → queue='{cfg.queue}'")

    def get_producer(self, producer_type: Any) -> Publisher:
        """등록된 Publisher를 반환합니다."""
        publisher = self._producers.get(producer_type)
        if publisher is None:
            raise KeyError(
                f"등록되지 않은 producer_type: {producer_type!r}. "
                "register_handlers()에서 먼저 등록하세요."
            )
        return publisher

    # ------------------------------------------------------------------
    # Consumer
    # ------------------------------------------------------------------

    def add_consumer(self, consumer: BaseConsumer) -> None:
        """소비자를 목록에 추가합니다."""
        self._consumers.append(consumer)

    async def start_consumers(self) -> None:
        """등록된 모든 소비자를 시작합니다."""
        if self._channel is None:
            raise RuntimeError("init()을 먼저 호출하세요.")

        for consumer in self._consumers:
            if not consumer.queue:
                logger.warning(
                    f"{type(consumer).__name__}: queue 이름이 비어 있어 건너뜁니다."
                )
                continue
            await self._channel.set_qos(prefetch_count=consumer.prefetch_count)
            queue = await self._channel.declare_queue(
                consumer.queue, durable=consumer.durable
            )
            await queue.consume(consumer._on_message)
            logger.info(
                f"Consumer 시작: {type(consumer).__name__} → queue='{consumer.queue}'"
            )
