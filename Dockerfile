# MailSort — Enterprise Email Triage OpenEnv Environment
# -------------------------------------------------------
# Build:  docker build -t mailsort-env .
# Run:    docker run -p 7860:7860 mailsort-env
# Test:   curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{}'

FROM python:3.11-slim

# Create non-root user (HuggingFace Spaces requirement)
RUN useradd -m -u 1000 user

USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

COPY --chown=user requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY --chown=user models.py    ./
COPY --chown=user openenv.yaml ./
COPY --chown=user server/      ./server/

EXPOSE 7860

# Healthcheck using Python (no curl needed)
HEALTHCHECK --interval=15s --timeout=10s --start-period=40s --retries=5 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health', timeout=8)" || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
