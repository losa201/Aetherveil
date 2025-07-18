# OSINT Agent Docker image
FROM aetherveil/base-agent:latest

# Install OSINT-specific tools
USER root
RUN apt-get update && apt-get install -y \
    theharvester \
    recon-ng \
    maltego \
    spiderfoot \
    shodan \
    whois \
    dnsrecon \
    fierce \
    && rm -rf /var/lib/apt/lists/*

# Install additional Python packages for OSINT
RUN pip install --no-cache-dir \
    shodan \
    censys \
    requests \
    beautifulsoup4 \
    tweepy \
    facebook-sdk \
    linkedin-api \
    instagram-private-api \
    google-api-python-client \
    dnspython \
    python-whois \
    haveibeenpwned \
    breach-parse

# Copy OSINT-specific configuration
COPY osint_config.json /app/config/

# Set environment variables
ENV AGENT_TYPE=osint
ENV AGENT_CAPABILITIES="social_media_intelligence,threat_intelligence,dark_web_monitoring,breach_data_analysis,reputation_analysis,email_intelligence,domain_intelligence,person_intelligence,company_intelligence,ip_intelligence,certificate_intelligence,paste_monitoring"

# Switch back to non-root user
USER agent

# Health check specific to OSINT agent
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from agents.osint_agent import OSINTAgent; print('OSINT agent healthy')" || exit 1

# Run OSINT agent
CMD ["python", "-m", "agents.osint_agent", "${AGENT_ID:-osint-$(hostname)}"]