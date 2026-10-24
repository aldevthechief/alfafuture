# CPU image: the encoders run on CPU in this deployment (a 4 GB GPU does not fit them all),
# news vectors are precomputed in data/retrieval_v2; only queries are encoded at runtime.
FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 CUDA_VISIBLE_DEVICES=-1
RUN apt-get update && apt-get install -y --no-install-recommends g++ git && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt
COPY . .
CMD ["python", "-X", "utf8", "-m", "pytest", "test_offline_scripts.py", "-q"]
