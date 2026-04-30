# CHSA Triage API - HF Space Docker (GPU T4 small recommande pour vLLM)
# Port obligatoire HF Spaces: 7860
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/user/.cache/huggingface \
    PORT=7860 \
    USE_VLLM=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip git curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3

# User non-root requis par HF Spaces (UID 1000)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"
WORKDIR /app

COPY --chown=user requirements-deploy.txt ./requirements-deploy.txt
RUN pip install --user --upgrade pip \
    && pip install --user -r requirements-deploy.txt

COPY --chown=user src/ ./src/
COPY --chown=user configs/ ./configs/

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -fsS http://localhost:7860/health || exit 1

CMD ["python", "-m", "uvicorn", "src.deployment.serve:app", "--host", "0.0.0.0", "--port", "7860"]
