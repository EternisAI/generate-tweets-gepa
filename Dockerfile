FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install dependencies
RUN pip install -r requirements.txt && \
    pip install dspy

# Set Python path
ENV PYTHONPATH=/app/src

# Command to run the worker
CMD ["python", "-m", "temporal_tweet_worker.worker"]
