import asyncio
import os
import time
from typing import List
import uuid

import pandas as pd

from kdb_manager import Document, UpsertDocument, upsert_documents


def get_chunk_id(ids: int) -> int:
    return f"index:{ids}_{time.perf_counter_ns()}"


async def main():
    file_path = "/Users/gridone/Documents/kac/학습데이터/한국공항공사 FAQ(배포).xlsx"
    df = pd.read_excel(file_path, header=0)

    datas = []

    for index, row in df.iterrows():
        query = row.iloc[1]  # 첫 번째 컬럼의 값
        answer = row.iloc[2]
        class_name = row.iloc[3]

        payload = {
            "context": query,
            "ids": get_chunk_id(index),
            "page_number": -1,
            "size": len(query),
            "metadatas": {
                "class": class_name,
                "paragraph_id": "",
                "paragraph_type": "faq",
                "doc_id": str(uuid.uuid4()),
                "file_name": os.path.basename(file_path),
                "data": answer,
            },
        }

        datas.append(payload)

    docs = {"collection_name": "kac", "documents": datas}

    res = await upsert_documents(UpsertDocument.model_validate(docs))
    print(res)


if __name__ == "__main__":
    asyncio.run(main())
