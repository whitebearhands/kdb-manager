"""
wrapper_redis
=============
asyncio 기반 Redis 클라이언트 래퍼.

- JSON 직렬화/역직렬화 내장
- 싱글톤: Redis() 는 항상 같은 인스턴스를 반환
- 의존: redis[asyncio]  (pip install redis[asyncio])
"""

import json
from typing import Any, Dict, List, Optional

try:
    import redis.asyncio as aioredis
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "wrapper_redis requires 'redis[asyncio]'. "
        "Install it with: pip install redis[asyncio]"
    ) from exc


def _serialize(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _deserialize(raw: Optional[str]) -> Any:
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw


class Redis:
    """
    asyncio Redis 클라이언트 (싱글톤).

    사용 예::

        client = Redis()
        await client.init(host="127.0.0.1", port=6379, db=0)

        await client.set("key", {"hello": "world"})
        data = await client.get("key")   # -> {"hello": "world"}

        await client.close()
    """

    _instance: Optional["Redis"] = None
    _client: Optional[aioredis.Redis] = None

    def __new__(cls) -> "Redis":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    async def init(
        self,
        host: str = "127.0.0.1",
        port: int = 6379,
        db: int = 0,
        username: Optional[str] = None,
        password: Optional[str] = None,
        ssl: bool = False,
        **_kwargs: Any,  # 알 수 없는 설정 키 무시
    ) -> None:
        scheme = "rediss" if ssl else "redis"
        url = f"{scheme}://{host}:{port}/{db}"
        self._client = await aioredis.from_url(
            url,
            username=username,
            password=password,
            decode_responses=True,
        )

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------

    def _ensure_connected(self) -> aioredis.Redis:
        if self._client is None:
            raise RuntimeError(
                "Redis 클라이언트가 초기화되지 않았습니다. init()을 먼저 호출하세요."
            )
        return self._client

    # ------------------------------------------------------------------
    # 단건 조작
    # ------------------------------------------------------------------

    async def get(self, key: str) -> Any:
        """키에 해당하는 값을 반환합니다. 없으면 None."""
        raw = await self._ensure_connected().get(key)
        return _deserialize(raw)

    async def set(self, key: str, value: Any, expire: Optional[int] = None) -> None:
        """값을 저장합니다. expire(초) 지정 시 TTL 설정."""
        await self._ensure_connected().set(key, _serialize(value), ex=expire)

    async def exists(self, key: str) -> bool:
        """키 존재 여부를 반환합니다."""
        return bool(await self._ensure_connected().exists(key))

    async def delete(self, key: str) -> None:
        """키를 삭제합니다."""
        await self._ensure_connected().delete(key)

    # ------------------------------------------------------------------
    # 다건 조작
    # ------------------------------------------------------------------

    async def mget(self, keys: List[str]) -> List[Any]:
        """여러 키의 값을 순서대로 반환합니다."""
        raws = await self._ensure_connected().mget(keys)
        return [_deserialize(r) for r in raws]

    async def set_all(self, data: Dict[str, Any], expire: Optional[int] = None) -> None:
        """딕셔너리의 모든 키-값을 파이프라인으로 저장합니다."""
        client = self._ensure_connected()
        async with client.pipeline(transaction=False) as pipe:
            for key, value in data.items():
                pipe.set(key, _serialize(value), ex=expire)
            await pipe.execute()

    async def get_all(self) -> List[str]:
        """현재 DB의 모든 키 목록을 반환합니다."""
        return await self._ensure_connected().keys("*")

    # ------------------------------------------------------------------
    # DB 초기화 / 연결 해제
    # ------------------------------------------------------------------

    async def clear(self) -> None:
        """현재 DB를 비웁니다 (FLUSHDB)."""
        await self._ensure_connected().flushdb()

    async def close(self) -> None:
        """연결을 닫습니다."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
