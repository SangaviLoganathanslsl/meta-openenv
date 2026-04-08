# ---------------------------------------------------------------------------
# Email Triage OpenEnv – Hugging Face Spaces compatible Docker image
# ---------------------------------------------------------------------------
FROM python:3.11-slim

# HF Spaces runs as a non-root user; create one
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY environment/ ./environment/
COPY app.py .
COPY openenv.yaml .
COPY inference.py .

# Ownership
RUN chown -R appuser:appuser /app

USER appuser

# HuggingFace Spaces expects port 7860
EXPOSE 7860

ENV PORT=7860
ENV PYTHONUNBUFFERED=1

CMD ["python", "app.py"]
