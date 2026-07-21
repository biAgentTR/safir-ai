# =============================================================================
# SAFIR - Production-grade container image
# NVIDIA CUDA 12.1 runtime tabanli, GPU/vLLM destekli calisma zamani.
# =============================================================================
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# --- Sistem bagimliliklari ---
# libgl1/libglib2.0-0: OpenCV'nin (headless olmayan) GUI baglantili kutuphane
# gereksinimlerini karsilar; python3.11: vLLM ve modern paketlerle uyum icin.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3-pip \
        python3.11-dev \
        libgl1 \
        libglib2.0-0 \
        ffmpeg \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

WORKDIR /app

COPY requirements.txt .
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install -r requirements.txt

COPY configs/ ./configs/
COPY src/ ./src/

RUN mkdir -p data outputs/reports outputs/timelines outputs/pdf

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python3", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
