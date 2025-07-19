# Discovery Agent Dockerfile
# Performs defensive reconnaissance and asset discovery

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    dnsutils \
    netcat-openbsd \
    nmap \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r discovery && useradd -r -g discovery -s /bin/bash discovery

# Copy requirements
COPY agents/discovery_agent/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY agents/discovery_agent/ .

# Set ownership
RUN chown -R discovery:discovery /app

# Switch to non-root user
USER discovery

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Run the agent
CMD ["python", "main.py"]