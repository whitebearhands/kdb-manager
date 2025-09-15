import json
from openai import AsyncOpenAI
from config import settings
from logging import getLogger

logger = getLogger(__name__)


async def generate_query_from_chunk(chunk) -> str:
    prompt = """주어진 Chunk의 내용을 기반으로 1개의 검색 가능한 질문을 생성해주세요. 
    질문은 검색 시스템이 해당 chunk를 정확하게 찾을 수 있도록 Chunk의 핵심 키워드와 주요 문구를 반드시 포함해야 합니다.
    
    질문 생성 시 다음 사항을 반드시 준수해주세요:
    1. 구체적인 정보 포함:
    - chunk의 핵심 키워드와 고유명사를 반드시 포함
    - 수치, 날짜, 이름 등 구체적 정보 활용
    - 테이블의 경우 컬럼명이나 주요 항목명을 포함
    2. 검색 가능성 최적화:
    - 원문의 주요 표현을 최대한 유지
    - 주요 키워드 혹은 키워드의 유사단어를 사용하여 질문을 작성
    - 추상적이거나 일반적인 표현 지양
    - 검색 시스템이 찾기 쉬운 구체적 표현 사용
    3. 질문 품질 기준:
    - 질문에 사용된 모든 주요 용어가 원문에 존재해야 함
    - key_terms에 포함된 모든 항목이 질문 내에서 사용되어야 함
    4. 실제 사람이 질문하는 내용처럼 간단하고 단순하게 작성되야 함
    - 반말, 줄임말등을 적극적으로 사용
    - 질문은 최대한 짧게 만들어야 함.
    
    출력은 다음 형식을 정확히 따라야 합니다:
    ```text
    질문 내용
    ```
    
    좋은 질문 예시:
    1. '2023년 클라우드 서비스 도입 비용은 얼마입니까?'
    - 구체적 수치와 날짜 포함
    - 원문의 표현을 그대로 활용
    - 명확한 답변이 가능한 형태
    2. '보안 정책 중 계정 관리 섹션에서 언급된 주요 지침은 무엇입니까?'
    - 명확한 위치와 컨텍스트 제공
    - 구체적인 섹션과 주제를 명시
    - 검색 가능한 키워드 포함
    
    피해야 할 질문 예시:
    1. '클라우드 서비스의 장점은 무엇인가요?'
    - 너무 일반적이고 추상적임
    - 특정 chunk를 지정하기 어려움
    2. '어떤 보안 정책이 가장 중요합니까?'
    - 주관적인 판단이 필요함
    - 검색 시스템이 찾기 어려운 형태
    3. 발전통합운영시스템데이터품질관리기준에서 '일자' 컬럼의 데이터 타입은 무엇입니까?
    - 테이블의 모양을 이용한 질문
    - 고객의 입장에서는 알 필요가 없는 질문.
    - chunk 를 검색할 수 없는 질문
    
    [입력 chunk 의 내용]
    
    """
    llm_client = AsyncOpenAI(
        base_url=settings.llm.base_url,
        api_key=settings.llm.api_key,
        timeout=None,  # 타임아웃 무제한 설정
        max_retries=2,  # 재시도 횟수는 필요에 따라 설정
    )
    try:
        messages = [
            {
                "role": "system",
                "content": "문장에서 핵심 내용을 뽑아 시험 문제지를 작성하는데 탁월한 기능을 가진 AI 입니다.",
            },
            {"role": "user", "content": prompt + chunk},
        ]

        request_form = {
            "frequency_penalty": settings.llm.frequency_penalty,
            "max_tokens": 5000,
            "model": settings.llm.model,
            "presence_penalty": 0,
            "stream": False,
            "temperature": 0.6,
            "top_p": 1,
            "stop": "<|im_end|>",
            "messages": messages,
        }

        stream = await llm_client.chat.completions.create(**request_form)
        return (
            stream.choices[0]
            .message.content.strip()
            .replace("```text", "")
            .replace("```", "")
        )
    except Exception as e:
        print(str(e))
    finally:
        await llm_client.close()


async def query_normalizer(user_query):
    llm_client = AsyncOpenAI(
        base_url=settings.llm.base_url,
        api_key=settings.llm.api_key,
        timeout=None,  # 타임아웃 무제한 설정
        max_retries=2,  # 재시도 횟수는 필요에 따라 설정
    )
    try:
        messages = [
            {
                "role": "system",
                "content": "당신은 입력된 사용자 질의를 검색을 위해 최적화 한 문장으로 다시 작성하는데 탁월한 성능을 보여주는 AI 이다",
            },
            {
                "role": "user",
                "content": f"""당신은 쿼리 정규화 AI입니다. 사용자의 다양한 질문을 벡터 데이터베이스 검색에 최적화된, **간결하고 표준화된 형태**로 변환해야 합니다. 핵심 의도와 키워드만 추출하고, 불필요한 대화체나 수식어는 모두 제거합니다.

**다음 지침을 따르세요:**
1.  **핵심 키워드만 추출**: 질문의 핵심 의미를 담은 키워드만 남깁니다.
2.  **불필요한 요소 제거**: 인사말, 감탄사, 질문 형식, 대화체, 오타 등 검색에 방해가 되는 모든 요소를 제거합니다.
3.  **명사 위주, 간결한 문구**: 정규화된 쿼리는 명사 위주의 짧은 문구나 단어 조합이어야 합니다.
4.  **추가 정보 금지**: 정규화된 쿼리 외에 어떠한 추가 설명이나 답변도 덧붙이지 않습니다.

**예시:**

* **사용자 쿼리**: "안녕하세요, 혹시 아이폰 15 프로 모델의 사양에 대해 알려주실 수 있으세요?"
* **정규화된 쿼리**: "아이폰 15 프로 사양"

* **사용자 쿼리**: "아, 제가 집에서 키우는 강아지가 갑자기 설사를 하는데, 병원에 가야 할까요?"
* **정규화된 쿼리**: "강아지 설사"

* **사용자 쿼리**: "서울에서 가장 좋은 데이트 장소 좀 추천해 주세요. 분위기 좋은 곳이면 좋겠어요."
* **정규화된 쿼리**: "서울 데이트 장소 추천"

* **사용자 쿼리**: "이번에 새로 나온 삼성 갤럭시 S24 울트라 모델 카메라가 어떻대요?"
* **정규화된 쿼리**: "갤럭시 S24 울트라 카메라"

---

**사용자 쿼리:** "{user_query}"
**정규화된 쿼리:**
""",
            },
        ]

        request_form = {
            "frequency_penalty": settings.llm.frequency_penalty,
            "max_tokens": 3000,
            "model": settings.llm.model,
            "presence_penalty": 0,
            "stream": False,
            "temperature": 0.6,
            "top_p": 1,
            "stop": "<|im_end|>",
            "messages": messages,
        }

        stream = await llm_client.chat.completions.create(**request_form)
        return stream.choices[0].message.content.strip()
    except Exception as e:
        print(str(e))
    finally:
        await llm_client.close()
