# ============================================================
# AI_2_CHALLENGE 팀 공용 개발 환경
# Base : PyTorch 2.7 + CUDA 12.8 (RTX 50xx / Blackwell 대응, Ch.4~5와 동일 검증 base)
# 의존성은 pyproject.toml 하나만 기준으로 삼는다 (버전 드리프트 방지)
# ============================================================
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Seoul \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git curl wget tzdata libgl1 libglib2.0-0 gcc g++ && \
    ln -sf /usr/share/zoneinfo/$TZ /etc/localtime && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# ── 의존성 레이어 캐싱 ──────────────────────────────────────
# pyproject.toml이 packages=["src","src.models"]를 선언하므로 빈 스텁을 먼저 만들어
# editable install(-e .)을 캐싱한다. 실제 소스는 아래에서 COPY로 덮어써도
# editable install은 경로만 참조하므로 계속 유효하다.
COPY pyproject.toml ./
RUN mkdir -p src/models && \
    touch src/__init__.py src/models/__init__.py

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -e . && \
    pip install --no-cache-dir jupyterlab ipykernel ipywidgets && \
    pip install --no-cache-dir flash-linear-attention==0.5.2

# base 이미지의 torchaudio는 torch 2.7.0 ABI로 빌드돼 있는데, pyproject.toml의
# torch>=2.11.0 요구사항 때문에 위에서 torch만 업그레이드되면서 ABI가 어긋난다
# (undefined symbol 에러로 transformers.audio_utils가 죽음). 프로젝트는 오디오를
# 쓰지 않으므로 torchaudio를 아예 제거해 이 충돌을 없앤다.
RUN pip uninstall -y torchaudio

# 실제 소스 코드 복사 (변경돼도 위 의존성 레이어는 재사용됨)
COPY . .
RUN pip install --no-cache-dir --no-deps -e .

EXPOSE 8888
CMD ["bash", "-lc", "jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --ServerApp.token=${JUPYTER_TOKEN:?JUPYTER_TOKEN is required} --ServerApp.allow_origin='*'"]
