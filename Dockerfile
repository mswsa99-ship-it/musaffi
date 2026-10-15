FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    YOLO_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    MPLCONFIGDIR=/tmp/mpl \
    YOLO_CONFIG_DIR=/tmp/yolo

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg git libgl1 libglib2.0-0 build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt \
    && pip install --no-deps deepfilternet==0.5.6 \
    && pip uninstall -y opencv-python 2>/dev/null; pip install --force-reinstall --no-deps opencv-python-headless

COPY app ./app
COPY scripts ./scripts

# النماذج تُحمَّل وقت بناء الصورة، فلا يحتاج الموقع أي اتصال خارجي أثناء التشغيل
ARG WITH_HQ=0
RUN if [ "$WITH_HQ" = "1" ]; then python scripts/download_models.py --hq; else python scripts/download_models.py; fi

RUN useradd -m musaffi && mkdir -p /app/data && chown -R musaffi /app/data
USER musaffi
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-keep-alive", "30"]
