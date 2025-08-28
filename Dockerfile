FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV LANGCHAIN_TRACING_V2=true

# Expose port for health checks
EXPOSE 8000

# Command will be overridden by LangGraph Cloud
CMD ["python", "-m", "langgraph", "serve"]