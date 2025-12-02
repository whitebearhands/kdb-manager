# 베이스 이미지: NVIDIA CUDA 12.8.1 + cuDNN + Development Tools + UBI 9
FROM nvidia/cuda:12.8.1-cudnn-devel-ubi9

# 시스템 패키지 업데이트 및 Python 3.12, git 설치
# UBI 9은 dnf를 패키지 관리자로 사용합니다.
# git은 pyproject.toml에서 git 저장소를 참조할 경우를 대비해 설치합니다.
RUN dnf update -y && \
    dnf install -y python3.12 python3.12-pip git && \
    dnf clean all

# 시스템의 기본 python3 명령어를 3.12 버전으로 설정
RUN alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 2

# astral-sh의 공식 이미지를 사용하여 uv 설치
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# 작업 디렉토리 설정
WORKDIR /app

# 종속성 파일 복사
COPY pyproject.toml uv.lock ./

# uv를 사용하여 가상 환경(.venv)에 종속성 설치
# .venv 디렉토리가 없으면 uv가 자동으로 생성합니다.
RUN uv sync --frozen --no-dev

# 애플리케이션 소스 코드 복사
COPY . .

# 보안을 위해 non-root 사용자 생성
RUN groupadd -r appuser && useradd --create-home -r -g appuser appuser

# /app 디렉토리의 모든 파일 소유권을 appuser에게 부여
RUN chown -R appuser:appuser /app

# non-root 사용자로 전환
USER appuser

# PATH 환경 변수에 가상 환경의 bin 디렉토리를 추가
# 이렇게 하면 'python'만 입력해도 가상 환경의 파이썬이 실행됩니다.
ENV PATH="/app/.venv/bin:$PATH"