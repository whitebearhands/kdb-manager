from typing import List, Union
import uuid
from fastapi import APIRouter

from kdb_manager import CreateCollection
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    MultiVectorConfig,
    MultiVectorComparator,
    HnswConfigDiff,
)
from colpali_engine.models import ColQwen2, ColQwen2Processor
from config import settings
from logging import getLogger
from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image
import torch


logger = getLogger(__name__)
router = APIRouter()
client = AsyncQdrantClient(url=settings.qdrant.url, api_key=settings.qdrant.api_key)

colqwen_model = ColQwen2.from_pretrained(
    "vidore/colqwen2-v1.0-merged",
    torch_dtype=torch.bfloat16,
    device_map="cuda" if torch.cuda.is_available() else "cpu",
).eval()

colqwen_processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v1.0-merged")


def pdf_to_images(pdf_path: Union[str, Path]) -> List[Image.Image]:
    """PDF를 페이지별 이미지로 변환"""
    try:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        logger.info(f"Converting PDF to images: {pdf_path.name}")
        images = convert_from_path(str(pdf_path), dpi=200)

        logger.info(f"Converted {len(images)} pages")
        return images

    except Exception as e:
        logger.error(f"Failed to convert PDF: {e}")
        raise


def get_patches(image_size):
    return colqwen_processor.get_n_patches(
        image_size,
        patch_size=colqwen_model.patch_size,
        spatial_merge_size=colqwen_model.spatial_merge_size,
    )


def embed_and_mean_pool_batch(image_batch):
    # embed
    with torch.no_grad():
        processed_images = colqwen_processor.process_images(image_batch).to(
            colqwen_model.device
        )
        image_embeddings = colqwen_model(**processed_images)

    image_embeddings_batch = image_embeddings.cpu().float().numpy().tolist()

    # mean pooling
    pooled_by_rows_batch = []
    pooled_by_columns_batch = []

    for image_embedding, tokenized_image, image in zip(
        image_embeddings, processed_images.input_ids, image_batch
    ):
        x_patches, y_patches = get_patches(image.size)

        image_tokens_mask = tokenized_image == colqwen_processor.image_token_id

        image_tokens = image_embedding[image_tokens_mask].view(
            x_patches, y_patches, colqwen_model.dim
        )
        pooled_by_rows = torch.mean(image_tokens, dim=0)
        pooled_by_columns = torch.mean(image_tokens, dim=1)

        image_token_idxs = torch.nonzero(image_tokens_mask.int(), as_tuple=False)
        first_image_token_idx = image_token_idxs[0].cpu().item()
        last_image_token_idx = image_token_idxs[-1].cpu().item()

        prefix_tokens = image_embedding[:first_image_token_idx]
        postfix_tokens = image_embedding[last_image_token_idx + 1 :]

        # print(f"There are {len(prefix_tokens)} prefix tokens and {len(postfix_tokens)} in a {model_name} PDF page embedding")

        # adding back prefix and postfix special tokens
        pooled_by_rows = (
            torch.cat((prefix_tokens, pooled_by_rows, postfix_tokens), dim=0)
            .cpu()
            .float()
            .numpy()
            .tolist()
        )
        pooled_by_columns = (
            torch.cat((prefix_tokens, pooled_by_columns, postfix_tokens), dim=0)
            .cpu()
            .float()
            .numpy()
            .tolist()
        )

        pooled_by_rows_batch.append(pooled_by_rows)
        pooled_by_columns_batch.append(pooled_by_columns)

    return image_embeddings_batch, pooled_by_rows_batch, pooled_by_columns_batch


def upload_batch(
    original_batch,
    pooled_by_rows_batch,
    pooled_by_columns_batch,
    payload_batch,
    collection_name,
):
    try:
        client.upload_collection(
            collection_name=collection_name,
            vectors={
                "mean_pooling_columns": pooled_by_columns_batch,
                "original": original_batch,
                "mean_pooling_rows": pooled_by_rows_batch,
            },
            payload=payload_batch,
            ids=[str(uuid.uuid4()) for i in range(len(original_batch))],
        )
    except Exception as e:
        print(f"Error during upsert: {e}")


def batch_embed_query(quries: List[str]):
    """쿼리 배치 임베딩 생성 (표준 구현)"""
    try:
        logger.info(f"Processing query batch: {len(quries)} queries")

        with torch.no_grad():
            processed_queries = ColQwen2Processor.process_queries(quries).to(
                colqwen_model.device
            )

            query_embeddings_batch = colqwen_model(**processed_queries)

        result = query_embeddings_batch.cpu().float().numpy()
        logger.info(f"Generated query embeddings shape: {result.shape}")
        return result

    except Exception as e:
        logger.error(f"Failed to process query batch: {e}")
        raise


def process_query(query: str) -> torch.Tensor:
    """쿼리 텍스트를 임베딩으로 변환 (호환성 유지)"""
    try:
        logger.info(f"Processing query: {query}")

        # 표준 배치 처리 방식 사용
        query_embeddings = batch_embed_query([query])

        # 첫 번째 쿼리 결과 반환
        result = torch.tensor(query_embeddings[0])
        logger.info(f"Generated query embeddings shape: {result.shape}")
        return result

    except Exception as e:
        logger.error(f"Failed to process query: {e}")
        raise


@router.post("/api/v1/vision/collection", operation_id="create_vision_collection")
async def create_vision_collection(request: CreateCollection):
    is_exist = await client.collection_exists(
        collection_name=f"{request.collection_name}"
    )
    if not is_exist:
        await client.create_collection(
            collection_name=request.collection_name,
            vectors_config={
                "original": VectorParams(
                    size=colqwen_model.dim,
                    distance=Distance.COSINE,
                    multivector_config=MultiVectorConfig(
                        comparator=MultiVectorComparator.MAX_SIM
                    ),
                    hnsw_config=HnswConfigDiff(m=0),
                ),
                "mean_pooling_columns": VectorParams(
                    size=colqwen_model.dim,
                    distance=Distance.COSINE,
                    multivector_config=MultiVectorConfig(
                        comparator=MultiVectorComparator.MAX_SIM
                    ),
                ),
                "mean_pooling_rows": VectorParams(
                    size=colqwen_model.dim,
                    distance=Distance.COSINE,
                    multivector_config=MultiVectorConfig(
                        comparator=MultiVectorComparator.MAX_SIM
                    ),
                ),
            },
        )

    return {"result": "Create collection successfully."}
