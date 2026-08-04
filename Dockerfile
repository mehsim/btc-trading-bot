# Use lightweight official Python runtime as a parent image
FROM python:3.10-slim

# Install system dependencies needed for compiling packages or downloading models
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set up user and home directory for Hugging Face Spaces (runs as non-root user 1000)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

ENV PYTHONUNBUFFERED=1
WORKDIR $HOME/app

# Copy dependency requirements first to leverage Docker build cache
COPY --chown=user requirements.txt $HOME/app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY --chown=user . $HOME/app

# Production web app port
ENV PORT=5001
EXPOSE 5001

# Run the trading bot main entrypoint
CMD ["python", "main.py"]
