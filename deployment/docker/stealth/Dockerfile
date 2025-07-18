# Stealth Agent Docker image
FROM aetherveil/base-agent:latest

# Install stealth-specific tools
USER root
RUN apt-get update && apt-get install -y \
    tor \
    proxychains \
    openvpn \
    stunnel \
    obfs4proxy \
    privoxy \
    polipo \
    socat \
    && rm -rf /var/lib/apt/lists/*

# Install additional Python packages for stealth operations
RUN pip install --no-cache-dir \
    stem \
    pysocks \
    requests \
    aiohttp \
    scapy \
    cryptography \
    pyopenssl \
    python-tor \
    fake-useragent \
    user-agents

# Copy stealth-specific configuration
COPY stealth_config.json /app/config/
COPY torrc /etc/tor/

# Set environment variables
ENV AGENT_TYPE=stealth
ENV AGENT_CAPABILITIES="traffic_obfuscation,proxy_management,evasion_techniques,anti_detection,behavior_mimicry,tor_routing,vpn_management,packet_crafting,timing_manipulation,fingerprint_spoofing,decoy_traffic,session_management"

# Configure Tor
RUN mkdir -p /var/lib/tor && \
    chown -R agent:agent /var/lib/tor && \
    chmod 700 /var/lib/tor

# Switch back to non-root user
USER agent

# Health check specific to stealth agent
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from agents.stealth_agent import StealthAgent; print('Stealth agent healthy')" || exit 1

# Run stealth agent
CMD ["python", "-m", "agents.stealth_agent", "${AGENT_ID:-stealth-$(hostname)}"]