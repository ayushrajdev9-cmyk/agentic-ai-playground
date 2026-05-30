FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN pip install --no-cache-dir -e ".[all]"

COPY config/ ./config/
COPY examples/ ./examples/

EXPOSE 8000

ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO

ENTRYPOINT ["agentic"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
