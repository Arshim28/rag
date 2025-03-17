FROM python:3.13-slim-bookworm AS builder

# Install uv from the official distroless image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency files first to leverage Docker caching
COPY pyproject.toml /app/
COPY uv.lock /app/

# Install dependencies without installing the project itself
RUN uv sync --frozen --no-install-project

# Copy application code
COPY config.yaml /app/
COPY data_processing/ /app/data_processing/
COPY embedding/ /app/embedding/
COPY storage/ /app/storage/
COPY retrieval/ /app/retrieval/
COPY utils/ /app/utils/
COPY rag_pipeline.py /app/
COPY main.py /app/

# Install the project
RUN uv sync --frozen

# Final stage
FROM python:3.13-slim-bookworm

# Copy the application and virtual environment from the builder stage
COPY --from=builder /app /app

# Set the virtual environment path
ENV PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Set the entrypoint
ENTRYPOINT ["uv", "run", "main.py"]