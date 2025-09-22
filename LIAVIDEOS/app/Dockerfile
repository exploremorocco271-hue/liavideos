# CUDA base for RunPod
FROM nvidia/cuda:12.1.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip \
    ffmpeg git curl ca-certificates build-essential \
 && rm -rf /var/lib/apt/lists/*

# Workdir
WORKDIR /app

# Copy files
COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --upgrade pip && pip install -r /app/requirements.txt

# Copy source
COPY utils.py transcribe.py features.py ranker.py renderer.py titles.py render_shorts.py api.py /app/
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

EXPOSE 8080
CMD ["/app/start.sh"]
