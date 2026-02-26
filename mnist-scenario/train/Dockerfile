FROM python:3.11-slim

# Avoid interactive prompts and keep image small-ish
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (tqdm and torch generally don't need much; kept minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train.py .

# Create dirs for mounted volumes (safe even if volumes override them)
RUN mkdir -p /app/data /app/outputs

CMD ["python", "train.py"]