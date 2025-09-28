FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# fara ENTRYPOINT; doar CMD neutru (va fi suprascris de "command" din compose)
CMD ["python", "--version"]
