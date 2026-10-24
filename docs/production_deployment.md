# TokenMesh Production Deployment Guide

This guide details best practices for deploying **TokenMesh** in production environments as a high-availability, low-latency AI Gateway and Reverse Proxy.

---

## 1. System Requirements

- **OS:** Linux (Ubuntu 22.04 LTS / Debian 12 / RHEL 9 recommended), macOS, or Windows Server.
- **CPU:** 2+ vCPUs (x86_64 or ARM64 with AVX2 / NEON for ONNX embeddings).
- **RAM:** Minimum 2 GB (4 GB recommended for high QPS and large HNSW graph cache).
- **Disk:** NVMe SSD recommended for fast DuckDB columnar read/write operations.

---

## 2. Docker & Docker Compose (Recommended)

### Production `docker-compose.yml`

```yaml
version: "3.8"

services:
  tokenmesh:
    image: tokenmesh:latest
    build:
      context: .
      dockerfile: Dockerfile
    container_name: tokenmesh-gateway
    restart: unless-stopped
    ports:
      - "8787:8787"
    environment:
      - HOST=0.0.0.0
      - PORT=8787
      - LOG_LEVEL=info
      - SIMILARITY_THRESHOLD=0.92
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - DUCKDB_PATH=/app/data/tokenmesh.duckdb
    volumes:
      # Persistent DuckDB storage volume
      - tokenmesh-data:/app/data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8787/healthz"]
      interval: 10s
      timeout: 3s
      retries: 3
      start_period: 5s

volumes:
  tokenmesh-data:
    driver: local
```

Deploying:
```bash
docker compose up -d --build
docker compose logs -f
```

---

## 3. Systemd Service (Bare Metal / VM)

Create `/etc/systemd/system/tokenmesh.service`:

```ini
[Unit]
Description=TokenMesh AI Gateway & Semantic Reverse Proxy
After=network.target

[Service]
Type=simple
User=tokenmesh
Group=tokenmesh
WorkingDirectory=/opt/tokenmesh
Environment="PATH=/opt/tokenmesh/.venv/bin"
EnvironmentFile=/opt/tokenmesh/.env
ExecStart=/opt/tokenmesh/.venv/bin/uvicorn core.main:app --host 0.0.0.0 --port 8787 --workers 2 --timeout-keep-alive 65
Restart=always
RestartSec=5
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now tokenmesh
sudo systemctl status tokenmesh
```

---

## 4. Nginx Reverse Proxy with TLS 1.3 & SSE Streaming

TokenMesh streams responses for both OpenAI (`stream=true`) and Anthropic (`event: content_block_delta`). Nginx must be configured to disable response buffering so streaming chunks pass through without delay.

Create `/etc/nginx/sites-available/tokenmesh.conf`:

```nginx
upstream tokenmesh_backend {
    server 127.0.0.1:8787;
    keepalive 32;
}

server {
    listen 443 ssl http2;
    server_name ai-gateway.yourcompany.internal;

    ssl_certificate     /etc/ssl/certs/tokenmesh.crt;
    ssl_certificate_key /etc/ssl/private/tokenmesh.key;
    ssl_protocols       TLSv1.2 TLSv1.3;
    ssl_ciphers         HIGH:!aNULL:!MD5;

    # Maximum client payload size (for long context prompts)
    client_max_body_size 64M;

    location / {
        proxy_pass http://tokenmesh_backend;
        proxy_http_version 1.1;

        # Keepalive headers
        proxy_set_header Connection "";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Disable buffering for SSE (Server-Sent Events) streaming
        proxy_buffering off;
        proxy_cache off;
        chunked_transfer_encoding on;

        # Extended timeouts for long reasoning models (o1, claude-3-7-sonnet)
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
    }
}
```

Test and reload Nginx:
```bash
sudo nginx -t && sudo systemctl reload nginx
```

---

## 5. Backup & Maintenance

### DuckDB Hot Backup
DuckDB supports safe checkpointing without downtime:
```bash
# Export analytics ledger and cache safely
duckdb data/tokenmesh.duckdb "EXPORT DATABASE 'backup_dir' (FORMAT PARQUET);"
```

### Pruning Expired Cache
Run the TokenMesh CLI purge command on a cron job:
```bash
# Keep last 30 days of semantic cache
0 3 * * * tokenmesh purge-cache --older-than 30 -y
```
