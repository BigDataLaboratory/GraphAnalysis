# Stage 1 — build
FROM python:3.11-slim AS builder
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake libxml2-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2 — runtime
FROM python:3.11-slim
WORKDIR /app

COPY requirements.txt .

COPY --from=builder /usr/local /usr/local

COPY Utils/       ./Utils/
COPY algorithms/  ./algorithms/
COPY community/   ./community/
COPY analysis/    ./analysis/
COPY properties/  ./properties/
COPY Main.py fetch_texts.py retweet.py ids_user.py config.py ./

VOLUME ["/app/resources", "/app/outputs", "/app/logs", "/app/properties"]
ENTRYPOINT ["python", "Main.py"]
ARG CACHE_BUST
RUN echo "Cache bust: $CACHE_BUST"