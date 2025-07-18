# Base Docker image for Aetherveil Sentinel agents
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    nmap \
    netcat \
    dnsutils \
    whois \
    curl \
    wget \
    tor \
    proxychains \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 agent && \
    mkdir -p /app/logs && \
    chown -R agent:agent /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set proper permissions
RUN chown -R agent:agent /app

# Switch to non-root user
USER agent

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/config

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV AGENT_LOG_LEVEL=INFO
ENV AGENT_DATA_DIR=/app/data
ENV AGENT_CONFIG_DIR=/app/config

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Default command (to be overridden by specific agent containers)
CMD ["python", "-m", "agents.base_agent"]