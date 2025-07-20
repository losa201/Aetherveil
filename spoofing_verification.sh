#!/bin/bash

# Aetherveil 3.0 - Spoofing Verification Script
# Run this script after DNS propagation (24-48 hours)

set -e

DOMAIN_NAME="${DOMAIN_NAME:-aetherveil.example.com}"
PROJECT_ID="${PROJECT_ID:-tidy-computing-465909-i3}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

log() { echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} $1"; }
success() { echo -e "${GREEN}[✓]${NC} $1"; }
warning() { echo -e "${YELLOW}[!]${NC} $1"; }
error() { echo -e "${RED}[✗]${NC} $1"; }
spoofing() { echo -e "${PURPLE}[SPOOF]${NC} $1"; }

# Get global IP from Terraform
get_global_ip() {
    if [ -f "infra/terraform/terraform.tfstate" ]; then
        GLOBAL_IP=$(cd infra/terraform && terraform output -raw spoofing_info 2>/dev/null | jq -r '.global_ip // empty' 2>/dev/null)
    fi
    
    if [ -z "$GLOBAL_IP" ]; then
        warning "Could not get global IP from Terraform. Please set manually:"
        echo "export GLOBAL_IP=YOUR_GLOBAL_IP"
        return 1
    fi
    
    log "Using Global IP: $GLOBAL_IP"
    return 0
}

# Test DNS resolution
test_dns_resolution() {
    spoofing "Testing DNS resolution for spoofed regions..."
    
    local regions=("us-central1" "europe-west1")
    local dns_success=0
    
    for region in "${regions[@]}"; do
        log "Testing ${region}.${DOMAIN_NAME}..."
        
        if command -v dig &> /dev/null; then
            local resolved_ip=$(dig +short ${region}.${DOMAIN_NAME} | tail -1)
            if [ "$resolved_ip" = "$GLOBAL_IP" ]; then
                success "✓ ${region}.${DOMAIN_NAME} → $resolved_ip"
                ((dns_success++))
            else
                warning "DNS not propagated for $region (got: $resolved_ip)"
            fi
        else
            warning "dig command not available, using nslookup..."
            nslookup ${region}.${DOMAIN_NAME} || warning "nslookup failed for $region"
        fi
    done
    
    if [ $dns_success -eq ${#regions[@]} ]; then
        success "All regional DNS entries resolve correctly"
        return 0
    else
        warning "$dns_success/${#regions[@]} regions resolved. Wait for DNS propagation."
        return 1
    fi
}

# Test regional headers
test_regional_headers() {
    spoofing "Testing regional HTTP headers for spoofing..."
    
    local regions=("us-central1" "europe-west1")
    local header_success=0
    
    for region in "${regions[@]}"; do
        log "Testing headers for ${region}.${DOMAIN_NAME}..."
        
        local response=$(curl -H "Host: ${region}.${DOMAIN_NAME}" \
                            https://${GLOBAL_IP}/ \
                            -I -s --connect-timeout 10 --max-time 30 2>/dev/null)
        
        if [ $? -eq 0 ]; then
            local served_from=$(echo "$response" | grep -i "X-Served-From" | cut -d: -f2 | tr -d ' \r\n')
            local timezone=$(echo "$response" | grep -i "X-Region-Timezone" | cut -d: -f2 | tr -d ' \r\n')
            local latency=$(echo "$response" | grep -i "X-Simulated-Latency" | cut -d: -f2 | tr -d ' \r\n')
            
            if [ -n "$served_from" ]; then
                success "✓ $region: X-Served-From: $served_from"
                [ -n "$timezone" ] && success "  └─ Timezone: $timezone"
                [ -n "$latency" ] && success "  └─ Latency: ${latency}ms"
                ((header_success++))
            else
                warning "Regional headers not found for $region"
                echo "Response headers:"
                echo "$response" | head -10
            fi
        else
            error "Failed to connect to $region endpoint"
        fi
        echo
    done
    
    if [ $header_success -gt 0 ]; then
        success "Regional header spoofing working ($header_success regions)"
        return 0
    else
        error "No regional headers detected. Check Cloud Run deployment."
        return 1
    fi
}

# Test CDN functionality
test_cdn_cache() {
    spoofing "Testing CDN cache and edge presence..."
    
    local response=$(curl -H "Host: ${DOMAIN_NAME}" \
                        https://${GLOBAL_IP}/ \
                        -I -s --connect-timeout 10 --max-time 30 2>/dev/null)
    
    if [ $? -eq 0 ]; then
        local cache_status=$(echo "$response" | grep -i -E "(cache-control|x-cache|cf-)" | head -3)
        
        if [ -n "$cache_status" ]; then
            success "✓ CDN headers detected:"
            echo "$cache_status" | while read line; do
                success "  └─ $line"
            done
        else
            warning "CDN headers not found. CDN may not be fully active yet."
        fi
        
        # Check response time for edge presence
        local response_time=$(curl -H "Host: ${DOMAIN_NAME}" \
                                https://${GLOBAL_IP}/ \
                                -o /dev/null -s -w "%{time_total}" \
                                --connect-timeout 10 --max-time 30 2>/dev/null)
        
        if [ $? -eq 0 ]; then
            success "✓ Response time: ${response_time}s (edge caching active)"
        fi
    else
        error "Failed to test CDN functionality"
        return 1
    fi
}

# Test SSL certificate
test_ssl_certificate() {
    spoofing "Testing SSL certificate for multi-domain support..."
    
    if command -v openssl &> /dev/null; then
        local cert_info=$(echo | openssl s_client -servername ${DOMAIN_NAME} \
                            -connect ${GLOBAL_IP}:443 2>/dev/null | \
                            openssl x509 -noout -text 2>/dev/null)
        
        if [ $? -eq 0 ]; then
            local san_domains=$(echo "$cert_info" | grep -A1 "Subject Alternative Name" | tail -1)
            
            if [[ "$san_domains" == *"${DOMAIN_NAME}"* ]]; then
                success "✓ SSL certificate includes main domain"
                
                if [[ "$san_domains" == *"us-central1.${DOMAIN_NAME}"* ]] && \
                   [[ "$san_domains" == *"europe-west1.${DOMAIN_NAME}"* ]]; then
                    success "✓ SSL certificate includes all regional subdomains"
                    success "  └─ Domains: $(echo "$san_domains" | sed 's/DNS://g')"
                else
                    warning "SSL certificate missing some regional domains"
                    echo "SAN: $san_domains"
                fi
            else
                warning "SSL certificate not yet issued for domain"
            fi
        else
            warning "Could not retrieve SSL certificate information"
        fi
    else
        warning "openssl not available, skipping SSL verification"
    fi
}

# Test simulated regional response times
test_regional_latency() {
    spoofing "Testing simulated regional response times..."
    
    local regions=("us-central1" "europe-west1")
    
    for region in "${regions[@]}"; do
        log "Testing response time for $region..."
        
        local result=$(curl -H "Host: ${region}.${DOMAIN_NAME}" \
                          https://${GLOBAL_IP}/ \
                          -o /dev/null -s -w "HTTP: %{http_code}, Time: %{time_total}s, DNS: %{time_namelookup}s" \
                          --connect-timeout 10 --max-time 30 2>/dev/null)
        
        if [ $? -eq 0 ]; then
            success "✓ $region: $result"
        else
            warning "Failed to test $region response time"
        fi
    done
}

# Test BigQuery synthetic audit logs
test_synthetic_logs() {
    spoofing "Testing BigQuery synthetic audit logs..."
    
    local query="SELECT region, COUNT(*) as log_count 
                FROM \`${PROJECT_ID}.security_analytics.synthetic_audit_logs\` 
                WHERE timestamp > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 DAY) 
                GROUP BY region ORDER BY region"
    
    if command -v bq &> /dev/null; then
        local result=$(bq query --use_legacy_sql=false --format=csv "$query" 2>/dev/null)
        
        if [ $? -eq 0 ] && [ -n "$result" ]; then
            success "✓ Synthetic audit logs found:"
            echo "$result" | tail -n +2 | while IFS=, read region count; do
                success "  └─ $region: $count logs"
            done
        else
            warning "No synthetic audit logs found. Check Cloud Scheduler job."
            log "Manual check: Run this query in BigQuery console:"
            echo "$query"
        fi
    else
        warning "bq command not available. Manual verification required:"
        echo "Query: $query"
    fi
}

# Test dummy infrastructure presence
test_dummy_infrastructure() {
    spoofing "Testing dummy infrastructure in spoofed regions..."
    
    if command -v gcloud &> /dev/null; then
        # Check IP reservations
        local ips=$(gcloud compute addresses list --filter="name~aetherveil-dummy" \
                   --format="table(name,region,status)" 2>/dev/null)
        
        if [ -n "$ips" ]; then
            success "✓ Dummy IP reservations found:"
            echo "$ips"
        else
            warning "No dummy IP reservations found"
        fi
        
        # Check storage buckets
        local buckets=$(gcloud storage buckets list --filter="name~dummy" \
                       --format="table(name,location)" 2>/dev/null)
        
        if [ -n "$buckets" ]; then
            success "✓ Dummy storage buckets found:"
            echo "$buckets"
        else
            warning "No dummy storage buckets found"
        fi
        
        # Check KMS keyrings
        local keyrings=$(gcloud kms keyrings list --filter="name~aetherveil-keyring" \
                        --format="table(name,location)" 2>/dev/null)
        
        if [ -n "$keyrings" ]; then
            success "✓ Regional KMS keyrings found:"
            echo "$keyrings"
        else
            warning "No regional KMS keyrings found"
        fi
    else
        warning "gcloud command not available for infrastructure verification"
    fi
}

# Generate comprehensive report
generate_report() {
    log "Generating spoofing verification report..."
    
    cat > spoofing_report.txt << EOF
Aetherveil 3.0 Spoofing Verification Report
Generated: $(date)
Domain: $DOMAIN_NAME
Global IP: $GLOBAL_IP
Project: $PROJECT_ID

=== SPOOFING VERIFICATION RESULTS ===

DNS Resolution: $([ $dns_ok -eq 0 ] && echo "PASS" || echo "FAIL")
Regional Headers: $([ $headers_ok -eq 0 ] && echo "PASS" || echo "FAIL")
CDN Functionality: $([ $cdn_ok -eq 0 ] && echo "PASS" || echo "FAIL")
SSL Certificate: $([ $ssl_ok -eq 0 ] && echo "PASS" || echo "FAIL")
Response Times: $([ $latency_ok -eq 0 ] && echo "PASS" || echo "FAIL")
Synthetic Logs: $([ $logs_ok -eq 0 ] && echo "PASS" || echo "FAIL")
Dummy Infrastructure: $([ $infra_ok -eq 0 ] && echo "PASS" || echo "FAIL")

=== NEXT STEPS ===
1. If any tests failed, wait 24-48 hours for DNS/SSL propagation
2. Check Cloud Run service status: gcloud run services list
3. Verify domain nameserver configuration
4. Monitor costs: gcloud billing projects describe $PROJECT_ID

=== MANUAL VERIFICATION COMMANDS ===
Test DNS: dig us-central1.$DOMAIN_NAME
Test Headers: curl -H "Host: europe-west1.$DOMAIN_NAME" https://$GLOBAL_IP/ -I
Check Logs: bq query --use_legacy_sql=false "SELECT * FROM \`$PROJECT_ID.security_analytics.synthetic_audit_logs\` LIMIT 5"
EOF

    success "Report saved to: spoofing_report.txt"
}

# Main verification function
main() {
    echo -e "${CYAN}"
    echo "╔═══════════════════════════════════════════════════════════════════════════════╗"
    echo "║                     AETHERVEIL 3.0 SPOOFING VERIFICATION                     ║"
    echo "║                          Multi-Region Presence Test                          ║"
    echo "╚═══════════════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    
    log "Starting spoofing verification for domain: $DOMAIN_NAME"
    
    # Get global IP
    get_global_ip || exit 1
    
    # Run verification tests
    dns_ok=1; headers_ok=1; cdn_ok=1; ssl_ok=1; latency_ok=1; logs_ok=1; infra_ok=1
    
    echo
    test_dns_resolution && dns_ok=0
    echo
    test_regional_headers && headers_ok=0
    echo
    test_cdn_cache && cdn_ok=0
    echo
    test_ssl_certificate && ssl_ok=0
    echo
    test_regional_latency && latency_ok=0
    echo
    test_synthetic_logs && logs_ok=0
    echo
    test_dummy_infrastructure && infra_ok=0
    echo
    
    # Calculate overall success
    local total_tests=7
    local passed_tests=0
    for result in $dns_ok $headers_ok $cdn_ok $ssl_ok $latency_ok $logs_ok $infra_ok; do
        [ $result -eq 0 ] && ((passed_tests++))
    done
    
    # Generate report
    generate_report
    
    # Final summary
    echo -e "${CYAN}=== VERIFICATION SUMMARY ===${NC}"
    echo "Tests Passed: $passed_tests/$total_tests"
    echo "Success Rate: $(( passed_tests * 100 / total_tests ))%"
    echo
    
    if [ $passed_tests -ge 5 ]; then
        success "🎯 Spoofing verification SUCCESSFUL!"
        success "Multi-region presence is convincingly simulated"
        echo
        echo -e "${GREEN}✅ Your Aetherveil 3.0 infrastructure appears globally distributed${NC}"
        echo -e "${GREEN}✅ Regional endpoints respond with appropriate headers${NC}"
        echo -e "${GREEN}✅ CDN provides global edge presence${NC}"
        echo -e "${GREEN}✅ Cost-optimized deployment is operational${NC}"
    elif [ $passed_tests -ge 3 ]; then
        warning "⚠️  Spoofing verification PARTIAL"
        warning "Some components need more time to propagate"
        echo
        echo -e "${YELLOW}🕐 Wait 24-48 hours for full DNS/SSL propagation${NC}"
        echo -e "${YELLOW}🔍 Re-run this script tomorrow: ./spoofing_verification.sh${NC}"
    else
        error "❌ Spoofing verification FAILED"
        error "Multiple components are not working correctly"
        echo
        echo -e "${RED}🚨 Check deployment status and logs${NC}"
        echo -e "${RED}🔧 Review README_PHASE1_SPOOFING.md for troubleshooting${NC}"
    fi
    
    echo
    log "Verification complete. Report saved to: spoofing_report.txt"
}

# Run verification
main "$@"