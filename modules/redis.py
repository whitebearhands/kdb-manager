from typing import Dict, List
from wrapper.redis_wrapper import Redis
from config import settings
from modules.singleton_meta import SingletonMeta


class RedisManager(metaclass=SingletonMeta):
    def __init__(self):
        self.redis = Redis()

    async def init(self):
        await self.redis.init(
            host=settings.redis.host, port=settings.redis.port, db=settings.redis.db
        )

    async def set(self, key, value):
        await self.redis.set(key, value)

    async def mset(self, key, value: Dict):
        await self.redis.set_all(value, expire=3600)

    async def get(self, key: str):
        return await self.redis.get(key)

    async def mget(self, key: List[str]):
        return await self.redis.mget(key)

    async def exists(self, key):
        return await self.redis.exists(key)

    async def remove(self, key):
        await self.redis.delete(key)

    async def remove_all(self):
        await self.redis.clear()

    async def get_all_keys(self):
        return await self.redis.get_all()
