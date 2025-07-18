# Reconnaissance Agent Docker image
FROM aetherveil/base-agent:latest

# Install reconnaissance-specific tools
USER root
RUN apt-get update && apt-get install -y \
    masscan \
    subfinder \
    amass \
    ffuf \
    gobuster \
    dirb \
    nikto \
    sqlmap \
    nuclei \
    httprobe \
    assetfinder \
    waybackurls \
    gau \
    && rm -rf /var/lib/apt/lists/*

# Install additional Python packages for reconnaissance
RUN pip install --no-cache-dir \
    python-nmap \
    dnspython \
    python-whois \
    requests \
    beautifulsoup4 \
    shodan \
    censys \
    socket \
    ssl

# Copy reconnaissance-specific configuration
COPY reconnaissance_config.json /app/config/

# Set environment variables
ENV AGENT_TYPE=reconnaissance
ENV AGENT_CAPABILITIES="dns_enumeration,port_scanning,service_detection,subdomain_discovery,whois_lookup,banner_grabbing,zone_transfer,reverse_dns"

# Switch back to non-root user
USER agent

# Health check specific to reconnaissance agent
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from agents.reconnaissance_agent import ReconnaissanceAgent; print('Reconnaissance agent healthy')" || exit 1

# Run reconnaissance agent
CMD ["python", "-m", "agents.reconnaissance_agent", "${AGENT_ID:-recon-$(hostname)}"]