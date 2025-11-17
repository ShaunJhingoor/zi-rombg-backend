FROM python:3.11-slim

# FFmpeg (Debian build includes zscale & prores_ks)
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN git clone https://github.com/PeterL1n/RobustVideoMatting.git /rvm
# Install deps first to leverage Docker layer cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App
COPY app.py .

# Default for local dev; Render will override PORT env
ENV PORT=8080
EXPOSE 8080

# 👇 change this line
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
