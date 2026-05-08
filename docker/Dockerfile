# Stage 1 — build C extensions (igraph, leidenalg)
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake libxml2-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2 — runtime image (no build tools)
FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY Utils/       ./Utils/
COPY algorithms/  ./algorithms/
COPY community/   ./community/
COPY analysis/    ./analysis/
COPY properties/  ./properties/
COPY Main.py fetch_texts.py retweet.py ids_user.py ./

# /app/resources → input data (read-only)
# /app/output    → analysis results
# /app/logs      → log files
# /app/properties → JSON config (override default properties/)
VOLUME ["/app/resources", "/app/output", "/app/logs", "/app/properties"]

ENTRYPOINT ["python", "Main.py"]
CMD ["--properties", "/app/properties/prop.json"]
