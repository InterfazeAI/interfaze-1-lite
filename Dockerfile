# Interfaze 1 Lite -- the whole model on one GPU, behind an OpenAI-compatible endpoint.
#
# Five processes share one card: vLLM serves the reasoning core, a second vLLM serves the
# document reader, a perception process holds the small models, a diarization process
# splits speakers, and the orchestrator answers /v1/chat/completions on :8000.
#
# Build:  docker build -t interfaze-1-lite .
# Run:    docker run --gpus all --ipc=host -p 8000:8000 -e HF_TOKEN=hf_... -v ./models:/models interfaze-1-lite
#
# HF_TOKEN is required: the diarization and guard models are gated on Hugging Face, and an
# anonymous download returns 401. Accept their terms on huggingface.co first.

FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3-pip git ffmpeg libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

# The same packages, pins and install order as the hosted deployment. vLLM, the perception
# models and the diarization pipeline co-install in one environment.
RUN python3.12 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
RUN pip install --no-cache-dir \
        "vllm==0.27.1" \
        "transformers==5.15.0" \
        "hf-transfer==0.1.9" \
        huggingface_hub accelerate "fastapi[standard]" python-multipart httpx pillow numpy soundfile \
        pyannote.audio \
        "paddleocr==3.3.2" "opencv-python-headless>=4.13.0" "paddlepaddle==3.2.2" \
        pymupdf \
        "sam-2 @ git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4" \
        einops peft "decord==0.6.0" "lmdb==1.7.5" \
        "nemo_toolkit[asr]"
RUN pip install --no-cache-dir "rapidfuzz==3.14.*"
RUN pip install --no-cache-dir pandas \
        "timesfm[torch] @ git+https://github.com/google-research/timesfm.git@bf88c5dc88f6275b7071ebcd9051406c11435c0c"

# Triton compiles a small C launcher the first time vLLM loads the model, and that needs
# Python's headers. Without them the reasoning core fails at startup with "Python.h: No
# such file or directory".
RUN apt-get update && apt-get install -y --no-install-recommends python3.12-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY src/interfaze_lite /app/interfaze_lite
COPY components.env.example /app/components.env.default
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV PYTHONPATH=/app \
    HF_HOME=/models \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    VLLM_CACHE_ROOT=/models/vllm \
    FLASHINFER_WORKSPACE_BASE=/models/vllm \
    TOKENIZERS_PARALLELISM=false \
    CUDA_HOME=/usr/local/cuda \
    TORCHINDUCTOR_COMPILE_THREADS=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    BRAIN_URL=http://127.0.0.1:8001 \
    PERCEPTION_URL=http://127.0.0.1:8002 \
    DIARIZE_URL=http://127.0.0.1:8003 \
    OCR_VLM_URL=http://127.0.0.1:8004 \
    BRAIN_SERVED_NAME=interfaze-lite \
    PERCEPTION_PRELOAD=asr_fallback,segmenter,guard,forecaster \
    MAX_TOOL_STEPS=8 \
    ENABLE_FAST_ASR=0 \
    BRAIN_SPEC_TOKENS=3

# Weights are a mounted volume, not a layer: ~50 GB baked into an image makes it
# unpullable and re-downloads everything on any code change.
VOLUME ["/models"]

EXPOSE 8000
# The first boot downloads ~50 GB of weights before the endpoint answers.
HEALTHCHECK --interval=30s --timeout=10s --start-period=3600s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

ENTRYPOINT ["/entrypoint.sh"]
