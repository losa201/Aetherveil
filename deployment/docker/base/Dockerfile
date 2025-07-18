# Multi-stage build for hardened Aetherveil Sentinel agents
# Stage 1: Build stage with full development tools
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Security tools stage
FROM python:3.11-slim as security-tools

# Install only essential security tools
RUN apt-get update && apt-get install -y \
    nmap \
    netcat-traditional \
    dnsutils \
    whois \
    curl \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Stage 3: Runtime stage with minimal footprint
FROM gcr.io/distroless/python3-debian11:nonroot

# Copy Python virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy security tools from security-tools stage
COPY --from=security-tools /usr/bin/nmap /usr/bin/nmap
COPY --from=security-tools /usr/bin/nc /usr/bin/nc
COPY --from=security-tools /usr/bin/dig /usr/bin/dig
COPY --from=security-tools /usr/bin/whois /usr/bin/whois
COPY --from=security-tools /usr/bin/curl /usr/bin/curl
COPY --from=security-tools /usr/bin/wget /usr/bin/wget
COPY --from=security-tools /etc/ssl /etc/ssl

# Copy application code
COPY --chown=nonroot:nonroot . /app

# Set working directory
WORKDIR /app

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV AGENT_LOG_LEVEL=INFO
ENV AGENT_DATA_DIR=/app/data
ENV AGENT_CONFIG_DIR=/app/config

# Create necessary directories as nonroot user
USER nonroot
RUN mkdir -p /app/data /app/logs /app/config

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Default command (to be overridden by specific agent containers)
CMD ["python", "-m", "agents.base_agent"]