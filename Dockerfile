FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git && \
    rm -rf /var/lib/apt-get/lists/*

# Install uv package manager
RUN pip install uv

# Copy project files
COPY pyproject.toml .
COPY src/ src/

# Install python packages into global system python
RUN uv pip install --system -e .

EXPOSE 7860

# Start FastAPI application on Hugging Face default port 7860
CMD ["uvicorn", "ragchat.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
