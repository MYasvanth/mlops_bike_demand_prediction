# MLOps Bike Demand Prediction Docker Image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed models/transformers reports/figures logs

# Set Python path
ENV PYTHONPATH=/app/src

# Expose ports for MLflow and API
EXPOSE 5000 8000

# Default command
CMD ["python", "run_pipeline.py"]