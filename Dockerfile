# Start from an official, lightweight Python image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /code

# Install system dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Add the /code directory to Python's import path
ENV PYTHONPATH=/code

# Copy and install Python dependencies first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

EXPOSE 8080

# Start app using Gunicorn with Uvicorn workers
CMD exec gunicorn -k uvicorn.workers.UvicornWorker app.main:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120 --log-level debug