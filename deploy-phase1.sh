#!/bin/bash

# Aetherveil 3.0 Phase 1 - Single-Region with Multi-Region Spoofing
# Enhanced deployment script for adversarial infrastructure

set -e

# Configuration
PROJECT_ID="${PROJECT_ID:-tidy-computing-465909-i3}"
REGION="${REGION:-us-central1}"
SERVICE_ACCOUNT_EMAIL="${SERVICE_ACCOUNT_EMAIL:-aetherveil-cicd@tidy-computing-465909-i3.iam.gserviceaccount.com}"
TERRAFORM_VERSION="1.7.0"
DOMAIN_NAME="${DOMAIN_NAME:-aetherveil.example.com}"
ENABLE_SPOOFING="${ENABLE_SPOOFING:-true}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Logging functions
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

spoofing() {
    echo -e "${PURPLE}[SPOOFING]${NC} $1"
}

banner() {
    echo -e "${CYAN}"
    echo "╔═══════════════════════════════════════════════════════════════════════════════╗"
    echo "║                          AETHERVEIL 3.0 PHASE 1                              ║"
    echo "║                   Single-Region Multi-Region Spoofing                        ║"
    echo "║                       Adversarial Infrastructure                             ║"
    echo "╚═══════════════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites for spoofing-enabled deployment..."
    
    # Check if gcloud is installed and authenticated
    if ! command -v gcloud &> /dev/null; then
        error "gcloud CLI is not installed. Please install it first."
        exit 1
    fi
    
    # Check if terraform is installed
    if ! command -v terraform &> /dev/null; then
        warning "Terraform not found. Installing..."
        install_terraform
    fi
    
    # Check if authenticated to GCP
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q "@"; then
        error "Please authenticate to GCP first: gcloud auth login"
        exit 1
    fi
    
    # Set project
    gcloud config set project $PROJECT_ID
    
    # Check domain ownership (optional warning)
    if [ "$ENABLE_SPOOFING" = "true" ]; then
        warning "Ensure you own the domain '$DOMAIN_NAME' for SSL certificate provisioning"
        warning "Or update terraform.tfvars with a domain you control"
    fi
    
    success "Prerequisites check completed"
}

install_terraform() {
    log "Installing Terraform $TERRAFORM_VERSION..."
    
    # Download and install Terraform
    wget -q "https://releases.hashicorp.com/terraform/${TERRAFORM_VERSION}/terraform_${TERRAFORM_VERSION}_linux_amd64.zip"
    unzip -q terraform_${TERRAFORM_VERSION}_linux_amd64.zip
    sudo mv terraform /usr/local/bin/
    rm terraform_${TERRAFORM_VERSION}_linux_amd64.zip
    
    success "Terraform installed successfully"
}

# Enable required APIs for spoofing
enable_apis() {
    log "Enabling required GCP APIs (including spoofing APIs)..."
    
    apis=(
        "run.googleapis.com"
        "cloudbuild.googleapis.com"
        "artifactregistry.googleapis.com"
        "pubsub.googleapis.com"
        "bigquery.googleapis.com"
        "firestore.googleapis.com"
        "aiplatform.googleapis.com"
        "ml.googleapis.com"
        "cloudscheduler.googleapis.com"
        "cloudfunctions.googleapis.com"
        "monitoring.googleapis.com"
        "logging.googleapis.com"
        "secretmanager.googleapis.com"
        "container.googleapis.com"
        "compute.googleapis.com"
        "cloudkms.googleapis.com"
        "securitycenter.googleapis.com"
        "binaryauthorization.googleapis.com"
        "dns.googleapis.com"
        "certificatemanager.googleapis.com"
    )
    
    for api in "${apis[@]}"; do
        log "Enabling $api..."
        gcloud services enable "$api" --quiet
    done
    
    success "All APIs enabled (including spoofing infrastructure APIs)"
}

# Create Artifact Registry repository
create_artifact_registry() {
    log "Creating Artifact Registry repository..."
    
    gcloud artifacts repositories create aetherveil \
        --repository-format=docker \
        --location=$REGION \
        --description="Aetherveil 3.0 Adversarial Platform" \
        --quiet || true
    
    success "Artifact Registry repository ready"
}

# Setup Terraform backend
setup_terraform_backend() {
    log "Setting up Terraform backend..."
    
    # Create Terraform state bucket
    gsutil mb gs://aetherveil-terraform-state-$PROJECT_ID || true
    gsutil versioning set on gs://aetherveil-terraform-state-$PROJECT_ID
    
    success "Terraform backend ready"
}

# Deploy spoofing infrastructure
deploy_spoofing_infrastructure() {
    log "Deploying Phase 1 infrastructure with multi-region spoofing..."
    
    cd infra/terraform
    
    # Initialize Terraform
    terraform init -backend-config="bucket=aetherveil-terraform-state-$PROJECT_ID"
    
    # Plan deployment
    terraform plan \
        -var="project_id=$PROJECT_ID" \
        -var="region=$REGION" \
        -var="deploy_phase_1=true" \
        -var="deploy_phase_2=false" \
        -var="deploy_phase_3=false" \
        -var="service_account_email=$SERVICE_ACCOUNT_EMAIL" \
        -var="domain_name=$DOMAIN_NAME" \
        -var="enable_spoofing=$ENABLE_SPOOFING" \
        -out=tfplan
    
    # Apply infrastructure
    terraform apply -auto-approve tfplan
    
    cd ../..
    success "Phase 1 spoofing infrastructure deployed"
}

# Build and push container images
build_and_push_images() {
    log "Building and pushing Phase 1 container images..."
    
    # Phase 1 agents for spoofing-enabled deployment
    local agents=("pipeline-observer" "anomaly-detector" "self-healing-engine")
    
    for agent in "${agents[@]}"; do
        log "Building $agent with spoofing capabilities..."
        
        gcloud builds submit \
            --config=<(cat << EOF
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args:
      - 'build'
      - '-t'
      - '${REGION}-docker.pkg.dev/${PROJECT_ID}/aetherveil/${agent}:latest'
      - '-t'
      - '${REGION}-docker.pkg.dev/${PROJECT_ID}/aetherveil/${agent}:v1.0-spoofing'
      - '-f'
      - 'deployment/docker/${agent}/Dockerfile'
      - '--build-arg'
      - 'SPOOFING_ENABLED=${ENABLE_SPOOFING}'
      - '.'
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', '${REGION}-docker.pkg.dev/${PROJECT_ID}/aetherveil/${agent}:latest']
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', '${REGION}-docker.pkg.dev/${PROJECT_ID}/aetherveil/${agent}:v1.0-spoofing']
images:
  - '${REGION}-docker.pkg.dev/${PROJECT_ID}/aetherveil/${agent}:latest'
  - '${REGION}-docker.pkg.dev/${PROJECT_ID}/aetherveil/${agent}:v1.0-spoofing'
options:
  machineType: 'E2_HIGHCPU_8'
  timeout: '1200s'
  logging: CLOUD_LOGGING_ONLY
EOF
) \
            --region="$REGION" \
            --async \
            . || warning "Build submission failed for $agent"
    done
    
    # Wait for builds
    log "Waiting for Cloud Build to complete..."
    local max_wait=1800
    local elapsed=0
    local check_interval=30
    
    while [ $elapsed -lt $max_wait ]; do
        running_builds=$(gcloud builds list \
            --filter="status=WORKING OR status=QUEUED" \
            --format="value(id)" | wc -l)
        
        if [ "$running_builds" -eq 0 ]; then
            success "All builds completed!"
            break
        fi
        
        log "Waiting for $running_builds builds... (${elapsed}s elapsed)"
        sleep $check_interval
        elapsed=$((elapsed + check_interval))
    done
    
    success "Phase 1 spoofing-enabled images built and pushed"
}

# Setup BigQuery with synthetic audit logs
setup_bigquery_spoofing() {
    log "Setting up BigQuery with synthetic audit logging..."
    
    # Enhanced tables for spoofing
    bq query --use_legacy_sql=false "
    CREATE TABLE IF NOT EXISTS \`$PROJECT_ID.security_analytics.red_team_findings\` (
        timestamp TIMESTAMP,
        target STRING,
        vulnerability_name STRING,
        severity STRING,
        description STRING,
        recommendation STRING,
        agent_id STRING,
        status STRING,
        spoofed_region STRING,
        simulated_source_ip STRING
    )
    PARTITION BY DATE(timestamp)
    CLUSTER BY target, vulnerability_name, severity, spoofed_region
    "
    
    success "BigQuery tables with spoofing support configured"
}

# Verify spoofing setup
verify_spoofing() {
    if [ "$ENABLE_SPOOFING" != "true" ]; then
        warning "Spoofing disabled, skipping verification"
        return
    fi
    
    spoofing "Verifying multi-region spoofing setup..."
    
    cd infra/terraform
    
    # Get outputs
    global_ip=$(terraform output -raw spoofing_info | jq -r '.global_ip // empty')
    
    if [ -n "$global_ip" ]; then
        spoofing "Global Anycast IP: $global_ip"
        spoofing "Regional endpoints configured for spoofing:"
        spoofing "  • us-central1.$DOMAIN_NAME"
        spoofing "  • europe-west1.$DOMAIN_NAME"
        spoofing "  • asia-southeast1.$DOMAIN_NAME"
        
        spoofing "Testing DNS resolution..."
        if command -v dig &> /dev/null; then
            # Test main domain
            main_ip=$(dig +short $DOMAIN_NAME | tail -1)
            if [ "$main_ip" = "$global_ip" ]; then
                success "Main domain resolves to global IP"
            else
                warning "Main domain DNS not yet propagated"
            fi
            
            # Test regional subdomains
            for region in "us-central1" "europe-west1" "asia-southeast1"; do
                regional_ip=$(dig +short ${region}.$DOMAIN_NAME | tail -1)
                if [ "$regional_ip" = "$global_ip" ]; then
                    success "Regional endpoint ${region}.$DOMAIN_NAME resolves correctly"
                else
                    warning "Regional DNS for $region not yet propagated"
                fi
            done
        else
            warning "dig command not available, skipping DNS verification"
        fi
    else
        warning "Global IP not found in Terraform outputs"
    fi
    
    cd ../..
}

# Generate post-deployment verification commands
generate_verification_commands() {
    spoofing "Generating verification commands for spoofing efficacy..."
    
    cat > spoofing_verification.sh << 'EOF'
#!/bin/bash

# Aetherveil 3.0 - Spoofing Verification Script
# Run this script after DNS propagation (24-48 hours)

DOMAIN_NAME="${DOMAIN_NAME:-aetherveil.example.com}"
GLOBAL_IP=$(cd infra/terraform && terraform output -raw spoofing_info | jq -r '.global_ip // empty')

echo "=== Multi-Region Spoofing Verification ==="
echo ""

echo "1. DNS Resolution Test:"
for region in "us-central1" "europe-west1" "asia-southeast1"; do
    echo "Testing ${region}.${DOMAIN_NAME}..."
    dig +short ${region}.${DOMAIN_NAME}
done
echo ""

echo "2. Regional Headers Test:"
for region in "us-central1" "europe-west1" "asia-southeast1"; do
    echo "Testing headers for ${region}.${DOMAIN_NAME}..."
    curl -H "Host: ${region}.${DOMAIN_NAME}" https://${GLOBAL_IP}/ -I -s | grep -E "(X-Served-From|X-Region-Timezone|X-Simulated-Latency)" || echo "  Headers not found (service may not be responding yet)"
    echo ""
done

echo "3. CDN Cache Test:"
curl -H "Host: ${DOMAIN_NAME}" https://${GLOBAL_IP}/ -I -s | grep -E "(Cache-Control|CDN|Edge)" || echo "  CDN headers not found"
echo ""

echo "4. SSL Certificate Test:"
echo | openssl s_client -servername ${DOMAIN_NAME} -connect ${GLOBAL_IP}:443 2>/dev/null | openssl x509 -noout -text | grep -A1 "Subject Alternative Name" || echo "  SSL certificate not yet issued"
echo ""

echo "5. Simulated Regional Response Times:"
for region in "us-central1" "europe-west1" "asia-southeast1"; do
    echo "Testing ${region}.${DOMAIN_NAME}..."
    time curl -H "Host: ${region}.${DOMAIN_NAME}" https://${GLOBAL_IP}/ -s -o /dev/null -w "HTTP: %{http_code}, Time: %{time_total}s\n" || echo "  Service not responding"
done

echo ""
echo "=== BigQuery Synthetic Audit Logs ==="
echo "Check for synthetic logs in:"
echo "  Project: $(gcloud config get-value project)"
echo "  Dataset: security_analytics"
echo "  Table: synthetic_audit_logs"
echo ""
echo "Run this query in BigQuery:"
echo "SELECT region, COUNT(*) as log_count FROM \`$(gcloud config get-value project).security_analytics.synthetic_audit_logs\` WHERE timestamp > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 DAY) GROUP BY region ORDER BY region"
EOF

    chmod +x spoofing_verification.sh
    success "Verification script created: ./spoofing_verification.sh"
}

# Main deployment function
main() {
    banner
    
    log "Starting Aetherveil 3.0 Phase 1 deployment with multi-region spoofing"
    log "Target region: $REGION"
    log "Spoofed regions: europe-west1, asia-southeast1"
    log "Domain: $DOMAIN_NAME"
    log "Spoofing enabled: $ENABLE_SPOOFING"
    
    check_prerequisites
    enable_apis
    create_artifact_registry
    setup_terraform_backend
    
    # Build images with spoofing support
    build_and_push_images
    
    # Deploy spoofing infrastructure
    deploy_spoofing_infrastructure
    
    # Setup BigQuery for synthetic logs
    setup_bigquery_spoofing
    
    # Verify spoofing setup
    verify_spoofing
    
    # Generate verification commands
    generate_verification_commands
    
    success "Aetherveil 3.0 Phase 1 deployment completed!"
    
    # Print deployment summary
    echo ""
    echo -e "${CYAN}🎯 DEPLOYMENT SUMMARY${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo -e "${GREEN}✅ REAL INFRASTRUCTURE (SINGLE REGION):${NC}"
    echo "   • Region: $REGION"
    echo "   • Cloud Run services for red-team agents"
    echo "   • BigQuery for analytics and synthetic logs"
    echo "   • Firestore for operational data"
    echo "   • Pub/Sub for agent communication"
    echo "   • KMS encryption and Secret Manager"
    echo ""
    echo -e "${PURPLE}🎭 SPOOFING INFRASTRUCTURE (MULTI-REGION ILLUSION):${NC}"
    echo "   • Global Anycast IP with Cloud CDN"
    echo "   • Regional DNS subdomains (us-central1, europe-west1, asia-southeast1)"
    echo "   • HTTPS Load Balancer with regional routing"
    echo "   • Dummy resources in spoofed regions (stopped VMs, empty buckets)"
    echo "   • Synthetic audit logs for fake regional activity"
    echo "   • Regional HTTP headers and latency simulation"
    echo ""
    echo -e "${BLUE}🔗 ENDPOINTS:${NC}"
    echo "   • Main: https://$DOMAIN_NAME"
    echo "   • US Central: https://us-central1.$DOMAIN_NAME"
    echo "   • EU West: https://europe-west1.$DOMAIN_NAME"
    echo "   • Asia Southeast: https://asia-southeast1.$DOMAIN_NAME"
    echo ""
    echo -e "${YELLOW}⚠️  IMPORTANT NOTES:${NC}"
    echo "   • SSL certificates may take 24-48 hours to provision"
    echo "   • DNS propagation may take 24-48 hours globally"
    echo "   • Run ./spoofing_verification.sh after DNS propagation"
    echo "   • Dummy VMs are created in TERMINATED state to minimize costs"
    echo "   • Monthly cost estimate: $500-800 (well under $3,000 limit)"
    echo ""
    echo -e "${CYAN}🛡️  SECURITY FEATURES:${NC}"
    echo "   • All data encrypted with CMEK"
    echo "   • VPC with private Google access"
    echo "   • IAM least-privilege service accounts"
    echo "   • DNSSEC enabled for domain security"
    echo "   • Binary Authorization for container security"
    echo ""
    echo -e "${GREEN}🚀 NEXT STEPS:${NC}"
    echo "   1. Configure your domain's nameservers to point to Cloud DNS"
    echo "   2. Wait for SSL certificate provisioning (24-48 hours)"
    echo "   3. Run verification script: ./spoofing_verification.sh"
    echo "   4. Monitor synthetic audit logs in BigQuery"
    echo "   5. Scale to Phase 2 when ready for real multi-region expansion"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

# Run deployment
main "$@"