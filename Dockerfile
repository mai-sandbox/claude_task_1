FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Expose port for LangGraph Studio
EXPOSE 8000

# Command for LangGraph deployment
CMD ["python", "-m", "langgraph", "serve", "--host", "0.0.0.0", "--port", "8000"]