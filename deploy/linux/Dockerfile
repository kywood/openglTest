# Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# OpenGL 오프스크린용 패키지들(xvfb + mesa)과 curl(헬스체크용)
RUN apt-get update && apt-get install -y --no-install-recommends \
    xvfb \
    xauth \
    mesa-utils \
    libgl1-mesa-dri \
    libgl1 \
    libglu1-mesa \
    libx11-6 \
    libxext6 \
    libxrandr2 \
    libxinerama1 \
    libxcursor1 \
    libxi6 \
    curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 앱 소스
COPY . .

EXPOSE 8000

# 간단 헬스체크 (문서 페이지가 뜨는지만 확인)
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
  CMD curl -fsS http://localhost:8000/docs > /dev/null || exit 1

# xvfb 가상디스플레이 위에서 uvicorn 실행
##CMD ["bash", "-lc", "xvfb-run -s \"-screen 0 1024x1024x24\" uvicorn gl1:app --host 0.0.0.0 --port 8000"]

CMD ["bash","-lc","Xvfb :99 -screen 0 1024x1024x24 & export DISPLAY=:99; exec uvicorn gl3:app --host 0.0.0.0 --port 8000"]
