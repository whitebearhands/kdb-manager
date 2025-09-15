import json
from typing import List, Dict, Any
import requests


def grid_rerank(query, fusion, rerank_top_k=5) -> List[Dict[str, Any]]:
    """
    벡터db 에서 검색된 결과를 사용자가 입력한 query 를 이용하여 리랭킹 한다.

    Args:
        query (str): 사용자 쿼리
        fusion (List[Dict[str, Any]]): 벡터DB 에서 검색한 결과
        rerank_top_k (int): 최대 응답 수

    Returns:
        List[Dict[str, Any]]: 검색된 청크 목록 (각 청크는 딕셔너리 형태)
    """
    url = "http://aiio.gridone.net:18080/api/new_reranker_ms/v2/predict"

    chunks = [
        {"context": result.payload["text"], "ids": result.payload["ids"]}
        for result in fusion
    ]

    packet = {
        "reqid": "97314",
        "input": {
            "items": [
                {
                    "query": query,
                    "k": rerank_top_k,
                    "retrieved_chunks": chunks,
                }
            ]
        },
    }

    payload = json.dumps(
        packet,
        ensure_ascii=False,
    )
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


def neural_search(
    collection_name: str,
    query: str,
    metadata_filter_key: str,
    match_values: List[str],
    top_k: int,
    use_paragraph: bool = False,
) -> List[Dict[str, Any]]:
    """
    사용자가 입력한 query 를 벡터DB 에서 검색하여 그 결과를 리턴한다.

    Args:
        collection_name (str): 검색할 벡터DB 의 컬렉션 이름
        query (str): 사용자 쿼리
        metadata_filter_key (str): 메타데이터를 이용해 필터 할 경우 메타데이터 내 키 값
        match_values (List[str]): 필터를 할 값 목록
        top_k (int): 최대 응답 수

    Returns:
        List[Dict[str, Any]]: 검색된 청크 목록 (각 청크는 딕셔너리 형태)
    """
    payload = {
        "collection_name": collection_name,
        "query": query,
        "query_normalize": False,
        "metadata_filter_key": metadata_filter_key,
        "match_values": match_values,
        "top_k": top_k,
        "use_paragraph": use_paragraph,
    }

    URI = "http://10.30.1.208:29911/api/v1/document/search_rerank"  # 임시로 유지
    res = requests.post(URI, json=payload)
    return res.json()


def query_normalizer(user_query):
    """
    사용자의 쿼리를 일반화 하여 검색에 적합한 새로운 쿼리를 생성한다. 특히 벡터검색, 리랭킹시에는 사용자 쿼리를 꼭 normalize 한 후 사용한다.

    Args:
        user_query (_type_): 사용자가 입력한 쿼리

    Returns:
        str: 새로 생성된 쿼리
    """
    payload = {
        "query": user_query,
    }

    URI = "http://10.30.1.208:29911/api/v1/query/normalize"  # 임시로 유지
    res = requests.post(URI, json=payload)
    return res.text
