# CPU image: the encoders run on CPU in this deployment (a 4 GB GPU does not fit them all),
# news vectors are precomputed in data/retrieval_v2; only queries are encoded at runtime.
FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=-1 \
    PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=10 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends g++ git && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    python -m pip install torch==2.6.0+cpu --index-url https://download.pytorch.org/whl/cpu
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    python -m pip install -r requirements.txt
COPY . .
# API services override this command; a one-off CLI container shows usage by default.
CMD ["python", "-X", "utf8", "pipeline.py", "--help"]
