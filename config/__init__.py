from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from typing import Literal, Optional

load_dotenv()


class microservice(BaseModel):
    reranker: str = Field(default="")


class Qdrant(BaseModel):
    url: str = Field(default="")
    api_key: str = Field(default="")


class EmbedingModel(BaseModel):
    dense_name: str = Field(default="")
    dense_path: str = Field(default="")
    sparse_name: str = Field(default="")
    sparse_path: str = Field(default="")
    model_device: str = Field(default="")
    rerank_name: str = Field(default="")
    rerank_path: str = Field(default="")


class AppSettings(BaseModel):
    port: int = Field(
        default=8100,
        ge=1,
        le=65_535,
        description="애플리케이션 포트 번호 (1~65535)",
    )
    service_name: str = Field(default="rag-manager", description="마이크로서비스 이름")
    group_name: Optional[str] = Field(
        default=None, description="마이크로서비스 그룹 이름"
    )
    eureka_server: str = Field(
        default="http://127.0.0.1:8761/eureka",
        description="서비스가 속한 존(Zone) 식별자",
    )
    use_ssl: bool = Field(default=False, description="HTTPS 사용 여부")
    use_management: bool = Field(default=False, description="관리 기능 사용 여부")
    site_key: Optional[str] = Field(
        default="",
        description="Site 키",
    )
    host_id: str = Field(
        default="1",
        description="호스트 ID",
    )
    host_ip: Optional[str] = Field(
        default=None,
        description="호스트 IP (Eureka 등록용)",
    )


class LogSettings(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="로그 레벨",
    )
    format: Literal["json", "plain"] = Field(
        default="plain",
        description="로그 포맷 (json 또는 plain)",
    )
    output_target: Literal["console", "file", "both"] = Field(
        default="both",
        description="로그 출력 대상",
    )
    dir: str = Field(
        default="./logs",
        description="로그 파일 디렉터리 경로",
        min_length=1,
    )
    filename: str = Field(
        default="app.log",
        description="로그 파일 디렉터리 경로",
        min_length=1,
    )
    file_max_bytes: int = Field(
        default=10 * 1024 * 1024,
        ge=1_024_000,  # 최소 1MB
        le=100 * 1024 * 1024,  # 최대 100MB
        description="로그 파일 최대 크기(바이트)",
    )
    file_backup_count: int = Field(
        default=7,
        ge=1,
        description="로그 파일 백업 개수",
    )


class RedisSettings(BaseModel):
    host: str = Field(
        default="127.0.0.1",
        description="Redis 호스트 주소",
    )
    port: int = Field(
        default=6379,
        ge=1,
        le=65_535,
        description="Redis 포트 (1~65535)",
    )
    db: int = Field(
        default=0,
        ge=0,
        le=15,
        description="Redis DB 번호 (0~15)",
    )
    username: Optional[str] = Field(
        default=None,
        description="Redis 인증 사용자 이름 (ACL 사용 시)",
    )
    password: Optional[str] = Field(
        default=None,
        description="Redis 인증 비밀번호",
    )
    ssl: bool = Field(
        default=False,
        description="SSL/TLS 사용 여부",
    )


class DBSettings(BaseModel):
    provider: Literal["mongo"] = Field(
        default="mongo",
        description="데이터베이스 프로바이더",
    )
    host: str = Field(
        default="127.0.0.1",
        description="데이터베이스 호스트 주소",
    )
    port: int = Field(
        default=27017,
        ge=1,
        le=65_535,
        description="데이터베이스 포트 (1~65535)",
    )
    username: Optional[str] = Field(
        default=None,
        description="데이터베이스 사용자 이름",
    )
    password: Optional[str] = Field(
        default=None,
        description="데이터베이스 비밀번호",
    )
    auth_source: Optional[str] = Field(
        default=None,
        description="MongoDB 인증 데이터베이스 (MongoDB 전용)",
    )


class MQSettings(BaseModel):
    provider: Literal["rabbitmq"] = Field(
        default="rabbitmq",
        description="AMQP 프로바이더",
    )
    host: str = Field(
        default="localhost",
        description="호스트 주소",
    )
    port: int = Field(
        default=5672,
        ge=1,
        le=65_535,
        description="포트 (1~65535)",
    )
    username: str = Field(
        default="guest",
        description="사용자 이름",
    )
    password: str = Field(
        default="guest",
        description="비밀번호",
    )
    virtual_host: str = Field(
        default="/",
        description="가상 호스트",
    )
    ssl: bool = Field(
        default=False,
        description="SSL/TLS 사용 여부",
    )
    heartbeat: int = Field(
        default=600,
        ge=0,
        description="하트비트 간격 (초)",
    )


class Settings(BaseSettings):
    runtime_env: Literal["prod", "dev"] = "prod"
    runtime_target: str = Field(
        default="default",
        description="런타임 타겟 (고객사별 설정 구분)",
    )

    app: AppSettings = Field(
        default_factory=AppSettings, description="Application 설정"
    )
    log: LogSettings = Field(
        default_factory=LogSettings,
        description="로그 설정",
    )
    redis: RedisSettings = Field(
        default_factory=RedisSettings,
        description="Redis 연결 및 캐시 설정",
    )
    db: DBSettings = Field(
        default_factory=DBSettings,
        description="데이터베이스 설정",
    )
    mq: MQSettings = Field(
        default_factory=MQSettings,
        description="AMQP 설정",
    )

    model: EmbedingModel = Field(
        default_factory=EmbedingModel,
        description="모델 설정",
    )
    qdrant: Qdrant = Field(
        default_factory=Qdrant,
        description="qdrant 설정",
    )
    ms: microservice = Field(
        default_factory=microservice,
        description="마이크로서비스 설정",
    )

    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"
        case_sensitive = False


settings = Settings()
