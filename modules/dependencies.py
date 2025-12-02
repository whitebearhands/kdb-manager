import time
from fastembed import SparseTextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from sentence_transformers import SentenceTransformer
import torch
from config import settings
import os
from logging import getLogger
from qdrant_client import AsyncQdrantClient
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = getLogger(__name__)

if torch.cuda.is_available():
    settings.model.model_device = "cuda"
elif torch.backends.mps.is_available():
    settings.model.model_device = "mps"
else:
    settings.model.model_device = "cpu"

settings.model.model_device = "cpu"

# Dense embedding model 설정
dense_model_path = os.path.join(settings.model.dense_path, settings.model.dense_name)
try:
    # 로컬 캐시 경로 지정
    embedding_model = SentenceTransformer(
        model_name_or_path=dense_model_path,
        device=settings.model.model_device,
        local_files_only=True,
    )
    logger.info(f"Using local dense model cache: {dense_model_path}")
except:
    # 온라인에서 다운로드
    logger.info(f"Downloading dense model from online: {settings.model.dense_name}")
    embedding_model = SentenceTransformer(
        model_name_or_path=settings.model.dense_name, device=settings.model.model_device
    )

# Sparse embedding model 설정
try:
    # 먼저 로컬 캐시로 시도
    sparse_embedding_model = SparseTextEmbedding(
        model_name=settings.model.sparse_name,
        cache_dir=settings.model.sparse_path,
        local_files_only=True,
    )
    logger.info(f"Using local sparse model cache: {settings.model.sparse_path}")
except:
    # 로컬에 없으면 다운로드
    logger.info(f"Downloading sparse model from online: {settings.model.sparse_name}")
    sparse_embedding_model = SparseTextEmbedding(
        model_name=settings.model.sparse_name,
        cache_dir=settings.model.sparse_path,
        local_files_only=False,
    )


client = AsyncQdrantClient(url=settings.qdrant.url, api_key=settings.qdrant.api_key)


def get_model_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def get_embedding_model() -> SentenceTransformer:
    return embedding_model


def get_sparse_model() -> SparseTextEmbedding:
    return sparse_embedding_model


def get_qdrant_client() -> AsyncQdrantClient:
    return client


def get_chunk_id() -> int:
    return time.perf_counter_ns()
