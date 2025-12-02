import aiohttp
import ssl
import numpy as np
from config import settings


async def get_connector():
    ssl_context = ssl.create_default_context()
    ssl_context.set_alpn_protocols(["h2", "http/1.1"])
    return aiohttp.TCPConnector(ssl_context=ssl_context, force_close=True)


async def encoding_documents(documents: list[str]):
    timeout = aiohttp.ClientTimeout(total=None)

    url = f"{settings.encoder.url}/api/v1/encoding_documents"
    # url = "http://80.188.223.202:16251/embed"
    async with aiohttp.ClientSession(
        connector=await get_connector(), timeout=timeout
    ) as session:
        # async with session.post(url, json={"documents": documents}) as resp:
        async with session.post(url, json={"documents": documents}) as resp:
            resp.raise_for_status()
            data = await resp.json()

            embeddings = data["embeddings"]  # List[List[float]]
            np_embeddings = np.array(embeddings)
            return np_embeddings


async def encoding_documents_fast(documents: list[str]):
    timeout = aiohttp.ClientTimeout(total=None)

    # url = "http://80.188.223.202:11316/embed"
    # url = "http://10.10.1.59:11316/embed"
    url = "http://10.10.1.64:18001/embed"
    async with aiohttp.ClientSession(
        connector=await get_connector(), timeout=timeout
    ) as session:
        # async with session.post(url, json={"documents": documents}) as resp:
        async with session.post(url, json={"inputs": documents}) as resp:
            resp.raise_for_status()
            data = await resp.json()
            np_embeddings = np.array(data)
            return np_embeddings


async def encoding_query(query: str):
    timeout = aiohttp.ClientTimeout(total=None)

    url = f"{settings.encoder.url}/api/v1/encoding_query"
    async with aiohttp.ClientSession(
        connector=await get_connector(), timeout=timeout
    ) as session:
        async with session.post(url, json={"query": query}) as resp:
            resp.raise_for_status()
            data = await resp.json()

            embeddings = data["embeddings"]  # List[List[float]]
            np_embeddings = np.array(embeddings)
            return np_embeddings


async def get_sentence_embedding_dimension() -> int:
    timeout = aiohttp.ClientTimeout(total=None)
    url = f"{settings.encoder.url}/api/v1/embedding_demension"
    async with aiohttp.ClientSession(
        connector=await get_connector(), timeout=timeout
    ) as session:
        async with session.get(url) as resp:
            resp.raise_for_status()
            data = await resp.json()

            return data["size"]
