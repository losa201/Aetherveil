# Scanner Agent Docker image
FROM aetherveil/base-agent:latest

# Install scanner-specific tools
USER root
RUN apt-get update && apt-get install -y \
    nikto \
    sqlmap \
    dirb \
    gobuster \
    ffuf \
    wpscan \
    nuclei \
    testssl.sh \
    sslscan \
    sslyze \
    && rm -rf /var/lib/apt/lists/*

# Install additional Python packages for scanning
RUN pip install --no-cache-dir \
    python-nmap \
    requests \
    beautifulsoup4 \
    selenium \
    aiohttp \
    lxml \
    urllib3 \
    sslyze \
    cryptography

# Copy scanner-specific configuration
COPY scanner_config.json /app/config/

# Set environment variables
ENV AGENT_TYPE=scanner
ENV AGENT_CAPABILITIES="vulnerability_scanning,web_scanning,network_scanning,ssl_analysis,configuration_analysis,cve_scanning,web_crawler,directory_bruteforce,injection_testing,authentication_testing"

# Switch back to non-root user
USER agent

# Health check specific to scanner agent
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from agents.scanner_agent import ScannerAgent; print('Scanner agent healthy')" || exit 1

# Run scanner agent
CMD ["python", "-m", "agents.scanner_agent", "${AGENT_ID:-scanner-$(hostname)}"]