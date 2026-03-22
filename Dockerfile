# IWSLT 2026 SST Submission -- NLLW AlignAtt SimulMT System
# CUNI (Charles University) -- Simultaneous Speech Translation
#
# Hardware: Single NVIDIA H100 80GB
# Protocol: SimulStream
# Metrics: LongYAAL (latency) + COMET wmt22-comet-da (quality, primary)
# Directions: EN-ZH, EN-DE, EN-IT, CS-EN
#
# Build:
#   docker build -t iwslt2026-cuni .
#
# Run (single direction):
#   docker run --gpus all iwslt2026-cuni \
#     --config configs/iwslt2026-en-zh.yaml
#
# Estimated image size: ~20 GB
# Estimated VRAM: ASR ~4GB + MT ~8GB = ~12GB total

# ===== Stage 1: Build llama.cpp with CUDA + attention extraction =====
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

RUN apt-get update && apt-get install -y \
    cmake build-essential git \
    && rm -rf /var/lib/apt/lists/*

# Build llama.cpp from source with CUDA support
# IMPORTANT: Must use the fork with attention extraction API (PR #20086)
COPY llama.cpp/ /build/llama.cpp/
WORKDIR /build/llama.cpp
RUN cmake -B build \
    -DGGML_CUDA=ON \
    -DLLAMA_CURL=OFF \
    -DBUILD_SHARED_LIBS=ON \
    && cmake --build build --config Release -j$(nproc)

# ===== Stage 2: Runtime image =====
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.12 python3.12-venv python3-pip \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf python3.12 /usr/bin/python3

WORKDIR /app

# Copy llama.cpp shared libraries from builder
COPY --from=builder /build/llama.cpp/build/src/libllama.so /app/lib/
COPY --from=builder /build/llama.cpp/build/ggml/src/libggml*.so /app/lib/
ENV LD_LIBRARY_PATH=/app/lib:${LD_LIBRARY_PATH}

# Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Models (large -- put early for Docker layer caching)
# ASR model: Qwen3-ASR-1.7B (~3.4 GB)
# MT model: HY-MT1.5-7B-Q8_0.gguf (~7.5 GB)
COPY models/ /app/models/

# Translation head configs
COPY nllw/heads/configs/ /app/heads/

# IWSLT 2026 per-direction configs
COPY configs/iwslt2026-*.yaml /app/configs/

# NLLW library
COPY nllw/ /app/nllw/
COPY setup.py pyproject.toml /app/
RUN pip3 install -e .

# Expose SimulStream HTTP API port
EXPOSE 8080

# Entrypoint: SimulStream HTTP server with our SpeechProcessor
# The evaluators connect to port 8080 via HttpProxySpeechProcessor
# For testing: override CMD with --test
ENTRYPOINT ["python3", "-m", "simulstream.server.http_speech_processor_server", \
    "--speech-processor", "nllw.simulstream:NLLWSpeechProcessor", \
    "--port", "8080"]
# Default config
CMD []
