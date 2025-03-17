FROM python:3.13-slim-bookworm AS builder

# Install build dependencies and Rust/Cargo
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    git \
    curl \
    ca-certificates \
    pkg-config \
    libssl-dev \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Rust and Cargo
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install uv using Cargo
RUN cargo install --git https://github.com/astral-sh/uv uv

WORKDIR /app

# Copy dependency files first to leverage Docker caching
COPY pyproject.toml /app/
COPY uv.lock /app/

# Now uv should be available in the PATH
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

# Final stage with minimal dependencies
FROM python:3.13-slim-bookworm

# Install packages needed for runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Copy uv from builder
COPY --from=builder /root/.cargo/bin/uv /usr/local/bin/uv

# Copy the application and virtual environment from the builder stage
COPY --from=builder /app /app
COPY --from=builder /app/.venv /app/.venv

WORKDIR /app

# Set the entrypoint
ENTRYPOINT ["uv", "run", "main.py"]