# KDB Manager

RAG(Retrieval-Augmented Generation) 파이프라인을 위한 벡터 DB 관리 서비스.

Qdrant 기반 하이브리드 검색(Dense + Sparse + RRF Fusion)과 외부 리랭커 연동,  
MongoDB 단락 원문 저장, RabbitMQ 비동기 임베딩 파이프라인을 제공한다.

---

## 목차

- [아키텍처](#아키텍처)
- [의존 서비스](#의존-서비스)
- [임베딩 모델](#임베딩-모델)
- [환경 설정](#환경-설정)
- [실행 방법](#실행-방법)
- [API 엔드포인트](#api-엔드포인트)
  - [컬렉션 관리](#컬렉션-관리)
  - [문서 관리](#문서-관리)
  - [검색](#검색)
  - [피드백](#피드백)
  - [쿼리 캐시](#쿼리-캐시)
- [데이터 모델](#데이터-모델)
- [검색 파이프라인](#검색-파이프라인)
- [임베딩 파이프라인 (MQ)](#임베딩-파이프라인-mq)
- [프로젝트 구조](#프로젝트-구조)

---

## 아키텍처

```
클라이언트
    │
    ▼
FastAPI (kdb_manager.py)
    ├── routes/collection.py   컬렉션 CRUD, 문서 조회
    ├── routes/document.py     업서트, 삭제, MongoDB 저장
    ├── routes/search.py       하이브리드 검색, 리랭킹, 단락 복원
    ├── routes/feedback.py     검색 피드백
    └── routes/query_cache.py  쿼리 캐싱
         │
         ├── Qdrant          벡터 DB (Dense + Sparse 벡터 저장/검색)
         ├── MongoDB         단락 원문 저장 (rag-data.documents, chunks.*)
         ├── RabbitMQ        비동기 임베딩 작업 큐
         ├── Redis           임베딩 작업 데이터 임시 저장
         └── Reranker API    외부 리랭킹 서비스
```

---

## 의존 서비스

| 서비스 | 용도 | 기본 포트 |
|--------|------|-----------|
| **Qdrant** | 벡터 저장 및 하이브리드 검색 | 6333 |
| **MongoDB** | 단락/청크 원문 저장 | 27017 |
| **RabbitMQ** | 비동기 임베딩 작업 큐 | 5672 |
| **Redis** | 임베딩 요청 데이터 임시 저장 | 6379 |
| **Reranker API** | 검색 결과 재순위화 | - |

---

## 임베딩 모델

| 종류 | 모델명 | 역할 |
|------|--------|------|
| **Dense** | `Qwen/Qwen3-Embedding-4B` | 의미 기반 벡터 검색 |
| **Sparse** | `Qdrant/bm42-all-minilm-l6-v2-attentions` | BM25 계열 키워드 검색 |
| **Reranker** | `Qwen/Qwen3-Reranker-4B` | 검색 결과 재순위화 (외부 서비스 사용) |

모델 파일은 `./models` 디렉토리에 저장한다. 없으면 서비스 시작 시 자동 다운로드.

---

## 환경 설정

루트의 `.env` 파일로 설정한다. `env.example`을 복사해 사용.

### APP

| 키 | 설명 | 예시 |
|----|------|------|
| `APP__PORT` | 서비스 포트 | `28101` |
| `APP__SERVICE_NAME` | 서비스 이름 (Eureka 등록용) | `kdb-manager` |
| `APP__EUREKA_SERVER` | Eureka 서버 주소 | `http://host:8761/eureka` |

### LOG

| 키 | 설명 | 기본값 |
|----|------|--------|
| `LOG__LEVEL` | 로그 레벨 (`DEBUG` \| `INFO` \| `WARNING` \| `ERROR`) | `INFO` |
| `LOG__FORMAT` | 출력 형식 (`json` \| `plain`) | `plain` |
| `LOG__OUTPUT_TARGET` | 출력 대상 (`console` \| `file` \| `both`) | `both` |
| `LOG__DIR` | 로그 파일 디렉토리 | `./logs` |
| `LOG__FILENAME` | 로그 파일 이름 | `app.log` |
| `LOG__FILE_MAX_BYTES` | 파일 최대 크기 (bytes) | `10485760` (10MB) |
| `LOG__FILE_BACKUP_COUNT` | 백업 파일 수 | `5` |

### REDIS

| 키 | 설명 | 기본값 |
|----|------|--------|
| `REDIS__HOST` | 호스트 | `127.0.0.1` |
| `REDIS__PORT` | 포트 | `6379` |
| `REDIS__DB` | DB 번호 | `0` |
| `REDIS__USERNAME` | 인증 사용자 (ACL) | - |
| `REDIS__PASSWORD` | 인증 비밀번호 | - |
| `REDIS__SSL` | SSL 사용 여부 | `false` |

### DB (MongoDB)

| 키 | 설명 | 기본값 |
|----|------|--------|
| `DB__HOST` | 호스트 | `127.0.0.1` |
| `DB__PORT` | 포트 | `27017` |
| `DB__USERNAME` | 사용자 이름 | - |
| `DB__PASSWORD` | 비밀번호 | - |
| `DB__AUTH_SOURCE` | 인증 DB | - |

### MQ (RabbitMQ)

| 키 | 설명 | 기본값 |
|----|------|--------|
| `MQ__HOST` | 호스트 | `localhost` |
| `MQ__PORT` | 포트 | `5672` |
| `MQ__USERNAME` | 사용자 이름 | `guest` |
| `MQ__PASSWORD` | 비밀번호 | `guest` |
| `MQ__VIRTUAL_HOST` | 가상 호스트 | `/` |
| `MQ__HEARTBEAT` | 하트비트 간격 (초) | `600` |

### MODEL

| 키 | 설명 |
|----|------|
| `MODEL__DENSE_NAME` | Dense 모델명 (HuggingFace ID 또는 로컬 경로) |
| `MODEL__DENSE_PATH` | Dense 모델 로컬 캐시 디렉토리 |
| `MODEL__SPARSE_NAME` | Sparse 모델명 |
| `MODEL__SPARSE_PATH` | Sparse 모델 로컬 캐시 디렉토리 |
| `MODEL__MODEL_DEVICE` | 실행 장치 (`cpu` \| `cuda` \| `mps`) |

### QDRANT

| 키 | 설명 |
|----|------|
| `QDRANT__URL` | Qdrant HTTP 주소 |
| `QDRANT__API_KEY` | API 키 (보안 설정 시) |

### 기타

| 키 | 설명 |
|----|------|
| `MS__RERANKER` | 외부 리랭커 서비스 Base URL |

---

## 실행 방법

### 로컬 실행

```bash
# 의존성 설치 (uv 사용)
uv sync

# 서버 시작
uvicorn kdb_manager:app --port 28101 --host 0.0.0.0

# 포트 오버라이드
python kdb_manager.py --port 28200
```

### Docker

```bash
# 이미지 빌드
docker build -t whitebearhands/kdb-manager:2.1.1 .

# Docker Compose 실행 (GPU 포함)
docker compose up -d
```

> `docker-compose.yml`은 `network_mode: host` + NVIDIA GPU 패스스루로 구성되어 있다.  
> `.env`, `./models`, `./logs` 디렉토리를 컨테이너에 마운트한다.

---

## API 엔드포인트

Swagger UI: `http://host:port/docs`

---

### 컬렉션 관리

#### `GET /api/v1/collection`
모든 Qdrant 컬렉션 목록을 반환한다.

**응답**
```json
[
  { "name": "my-collection", "status": "green" }
]
```

---

#### `POST /api/v1/collection`
컬렉션을 생성한다. 이미 존재하면 무시한다.

**요청 바디**
```json
{ "collection_name": "my-collection" }
```

**컬렉션 설정**
- Dense 벡터: COSINE 거리, HNSW (m=64, ef_construct=1000)
- Sparse 벡터: BM42 기반 (on_disk=false)
- 세그먼트 10개, 최적화 스레드 8개

**응답**
```json
{ "result": "Create collection successfully." }
```

---

#### `DELETE /api/v1/collection/{collection_name}`
컬렉션과 내부 모든 데이터를 영구 삭제한다.

> `collection_name`은 HTML 인코딩된 문자열로 전달한다.

**응답**
```json
{ "result": "Delete collection successfully." }
```

---

#### `GET /api/v1/documents/{collection_name}`
컬렉션의 포인트(청크)를 페이징으로 조회한다.

**쿼리 파라미터**

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `page` | int | 1 | 페이지 번호 |
| `page_size` | int | 10 | 페이지 크기 (최대 1000) |

**응답**
```json
{
  "page": [
    {
      "context": "청크 텍스트",
      "ids": "doc-id",
      "metadatas": { "file_name": "파일.pdf", "doc_id": "..." }
    }
  ],
  "page_info": {
    "total_elements": 1500,
    "total_pages": 150,
    "page": 1,
    "first": true,
    "last": false,
    "empty": false
  }
}
```

---

#### `GET /api/v1/{collection_name}/{p_id}/paragraph/content`
MongoDB에서 특정 단락의 원문을 단건 조회한다.

**응답**
```json
{
  "collection_id": "my-collection",
  "paragraph_id": "para-001",
  "context": "단락 원문 텍스트...",
  "metadatas": { "doc_id": "...", "file_name": "..." }
}
```

---

### 문서 관리

#### `POST /api/v1/documents`
문서 청크 목록을 임베딩 후 Qdrant에 upsert한다.

컬렉션이 없으면 자동 생성한다.

**요청 바디**
```json
{
  "collection_name": "my-collection",
  "documents": [
    {
      "context": "청크 본문 텍스트",
      "ids": "원본 문서 ID",
      "page_number": 3,
      "size": 512,
      "metadatas": {
        "file_name": "문서.pdf",
        "doc_id": "doc-001",
        "collection_id": "my-collection",
        "paragraph_id": "para-001"
      }
    }
  ]
}
```

**처리 과정**
1. Dense + Sparse 임베딩 병렬 생성 (asyncio.gather)
2. 적응형 배치 upsert (초기 100건, OOM 시 절반씩 감소)
3. 10% 샘플 무결성 검증
4. 실패 문서 자동 재시도 (50건 미만 시)

**응답**
```json
{
  "result": "95/100 items added successfully",
  "details": {
    "total_requested": 100,
    "successful": 95,
    "failed": 5,
    "integrity_verified": true,
    "integrity_rate": 0.97
  }
}
```

---

#### `POST /api/v1/mongo/paragraph`
단락 목록을 MongoDB `rag-data.documents`에 직접 저장한다.

**요청 바디**
```json
{
  "pages": [
    {
      "collection_id": "my-collection",
      "paragraph_id": "para-001",
      "context": "단락 원문...",
      "metadatas": { "doc_id": "doc-001", "file_name": "파일.pdf" }
    }
  ]
}
```

---

#### `DELETE /api/v1/file`
파일 이름으로 해당 파일의 모든 청크를 Qdrant에서 삭제한다.

**요청 바디**
```json
{
  "file_name": "삭제할파일.pdf",
  "collection_name": "my-collection"
}
```

---

#### `DELETE /api/v1/documents/{collection_name}/{id}`
문서 ID(`ids` 필드)로 포인트를 Qdrant에서 삭제한다.

> `collection_name`은 HTML 인코딩된 문자열로 전달한다.

---

### 검색

#### `POST /api/v2/document/search_rerank` ⭐ 메인 검색
하이브리드 검색 → 리랭킹 → (선택) 단락 복원 파이프라인.

**요청 바디**

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| `collection_name` | string | ✅ | 검색할 컬렉션 |
| `query` | string | ✅ | 검색 쿼리 |
| `top_k` | int | - | 검색 후보 수 (기본 100) |
| `use_paragraph` | bool | - | true면 MongoDB 단락 원문 반환 (기본 false) |
| `metadata_filter_key` | string | - | 메타데이터 필터 키 (예: `"doc_id"`) |
| `match_values` | string[] | - | 필터 값 목록 |
| `room_id` | string | - | 채팅방 단위 검색 범위 제한 |

**요청 예시**
```json
{
  "collection_name": "my-collection",
  "query": "RAG 파이프라인 구성 방법",
  "top_k": 50,
  "use_paragraph": true,
  "metadata_filter_key": "doc_id",
  "match_values": ["doc-001", "doc-002"]
}
```

**응답 (`use_paragraph=false`)**
```json
[
  {
    "context": "청크 텍스트",
    "ids": "doc-id",
    "metadatas": { "file_name": "...", "doc_id": "..." },
    "reranked_score": 0.92
  }
]
```

**응답 (`use_paragraph=true`)**
```json
[
  {
    "collection_id": "my-collection",
    "paragraph_id": "para-001",
    "context": "단락 원문 텍스트 (청크보다 더 긴 맥락)",
    "metadatas": { "doc_id": "doc-001", "file_name": "파일.pdf" },
    "rerank_score": 0.92
  }
]
```

---

#### `POST /api/v2/document/search_paragraph`
검색 + 리랭킹 후 `paragraph_id`와 점수만 반환한다.

단락 내용 없이 ID/점수만 필요한 경우 (프론트 lazy-load 등) 사용한다.

**요청 바디**: `search_rerank`와 동일

**응답**
```json
[
  { "paragraph_id": "para-001", "score": 0.92 },
  { "paragraph_id": "para-003", "score": 0.87 }
]
```

---

#### `POST /api/v1/document/search`
리랭킹 없는 순수 하이브리드 검색. Qdrant RRF 퓨전 결과를 그대로 반환한다.

**요청 바디**: `search_rerank`와 동일 (`use_paragraph`, `room_id` 미적용)

**응답**: Qdrant 포인트 목록 (score_threshold=0.11 적용)

---

## 데이터 모델

### Document (청크 단위 입력)

```python
class Document:
    context: str           # 청크 본문 텍스트
    ids: str               # 원본 문서 ID
    page_number: int = -1  # 원본 문서 페이지 번호 (-1: 알 수 없음)
    size: int              # 청크 글자 수
    metadatas: Dict        # 부가 정보 (아래 참조)
```

**metadatas 권장 필드**

| 필드 | 설명 |
|------|------|
| `file_name` | 원본 파일명 |
| `doc_id` | 문서 고유 ID |
| `collection_id` | 컬렉션 ID |
| `paragraph_id` | 단락 ID (MongoDB 조회 키) |
| `paragraph_type` | 단락 유형 (`"faq"` 이면 MongoDB 조회 생략) |
| `room_id` | 채팅방 ID (room_id 필터 사용 시) |
| `bbox` | 원본 문서 내 위치 정보 |

---

### Qdrant 포인트 페이로드

Qdrant에 저장되는 포인트의 payload 구조:

```json
{
  "context": "청크 텍스트",
  "ids": "원본 문서 ID",
  "page_number": 3,
  "size": 512,
  "metadatas": {
    "file_name": "문서.pdf",
    "doc_id": "doc-001",
    "collection_id": "my-collection",
    "paragraph_id": "para-001"
  }
}
```

---

### MongoDB 스키마

**`rag-data.documents`** — 단락 원문 저장

```json
{
  "collection_id": "my-collection",
  "paragraph_id": "para-001",
  "context": "단락 전체 원문",
  "metadatas": {
    "doc_id": "doc-001",
    "file_name": "문서.pdf",
    "bbox": [...]
  }
}
```

인덱스: `(collection_id, metadatas.doc_id, paragraph_id)` 복합 인덱스

**`chunks.<collection_name>`** — 청크 원문 저장 (컬렉션별 분리)

---

## 검색 파이프라인

```
쿼리 입력
   │
   ├─ Dense 임베딩 (Qwen3-Embedding-8B)
   └─ Sparse 임베딩 (BM42)
         │
         ▼
   Qdrant Prefetch (각각 top_k 후보 수집)
         │
         ▼
   RRF Fusion (두 결과 통합 → 50건)
         │
         ▼
   외부 리랭커 (Qwen3-Reranker-4B) → top 5
         │
         ▼
   [use_paragraph=true]
   MongoDB rag-data.documents 에서 단락 원문 조회
         │
         ▼
   최종 결과 반환
```

---

## 임베딩 파이프라인 (MQ)

RabbitMQ 기반 비동기 임베딩 처리 흐름:

```
외부 서비스
   │ rag.embedding.request 큐에 메시지 발행
   ▼
EmbeddingConsumer (kdb_manager.py)
   │
   ├─ 1. Redis에서 청크/단락 데이터 로드
   │      redis_chunk_key → {"data": [...chunks]}
   │      redis_para_key  → {"data": [...paragraphs]}
   │
   ├─ 2. POST /api/v1/documents (Qdrant upsert)
   │
   ├─ 3. chunk_to_mongo()   → MongoDB chunks.<collection_id>
   │
   ├─ 4. paragraph_to_mongo() → MongoDB rag-data.documents
   │
   └─ 5. MQ 이벤트 발행
          성공: rag.embedding.completed
          실패: rag.embedding.failed
```

**수신 메시지 형식**

```json
{
  "job_id": "unique-job-id",
  "file_id": "file-storage-id",
  "doc_id": "document-id",
  "collection_id": "my-collection",
  "redis_chunk_key": "rag.document.{doc_id}.chunk",
  "redis_para_key":  "rag.document.{doc_id}.para",
  "redis_img_key":   "rag.document.{doc_id}.img",
  "redis_tbl_key":   "rag.document.{doc_id}.tbl"
}
```

---

## 프로젝트 구조

```
kdb-manager/
├── kdb_manager.py          진입점 (앱, 라이프사이클, MQ Consumer)
├── routes/
│   ├── collection.py       컬렉션 CRUD, 문서 조회
│   ├── document.py         업서트, 삭제, MongoDB 저장
│   ├── search.py           하이브리드 검색, 리랭킹
├── modules/
│   ├── dependencies.py     모델/클라이언트 싱글톤
│   ├── redis.py            RedisManager
│   └── singleton_meta.py   싱글톤 메타클래스
├── config/
│   └── __init__.py         pydantic-settings 설정 모델
├── wrapper/
│   ├── logger_wrapper.py   로거 설정
│   ├── redis_wrapper.py    Redis 클라이언트
│   ├── rabbitmq_wrapper.py RabbitMQ 클라이언트
│   └── rabbitmq_wrapper_for_rag.py  RAG 전용 Producer/Consumer
├── models/                 임베딩 모델 파일 (git 미포함)
├── logs/                   로그 파일 (git 미포함)
├── pyproject.toml
├── docker-compose.yml
├── Dockerfile
└── .env                    환경 변수 (git 미포함)
```
