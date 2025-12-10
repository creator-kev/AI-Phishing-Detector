FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs models data

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Expose port (Railway will still set its own $PORT)
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"

# Start command (FIXED: uses shell so ${PORT} expands correctly on Railway)
CMD ["sh", "-c", "gunicorn app.main:app --workers 2 --bind 0.0.0.0:${PORT} --timeout 120 --access-logfile - --error-logfile - --log-level info"]
