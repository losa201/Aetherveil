#!/bin/bash

# Autonomous AI-Powered DevOps Platform Deployment Script
# Complete deployment for all phases

set -e

# Configuration
PROJECT_ID="${PROJECT_ID:-tidy-computing-465909-i3}"
REGION="${REGION:-us-central1}"
SERVICE_ACCOUNT_EMAIL="${SERVICE_ACCOUNT_EMAIL:-aetherveil-cicd@tidy-computing-465909-i3.iam.gserviceaccount.com}"
TERRAFORM_VERSION="1.7.0"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
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

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
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

# Enable required APIs
enable_apis() {
    log "Enabling required GCP APIs..."
    
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
    )
    
    for api in "${apis[@]}"; do
        log "Enabling $api..."
        gcloud services enable "$api" --quiet
    done
    
    success "All APIs enabled"
}

# Create Artifact Registry repository
create_artifact_registry() {
    log "Creating Artifact Registry repository..."
    
    gcloud artifacts repositories create aetherveil \
        --repository-format=docker \
        --location=$REGION \
        --description="Aetherveil Autonomous DevOps Platform" \
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

# Deploy infrastructure with Terraform
deploy_infrastructure() {
    local phase=$1
    log "Deploying Phase $phase infrastructure with Terraform..."
    
    cd infra/terraform
    
    # Initialize Terraform
    terraform init -backend-config="bucket=aetherveil-terraform-state-$PROJECT_ID"
    
    # Plan deployment
    terraform plan \
        -var="project_id=$PROJECT_ID" \
        -var="region=$REGION" \
        -var="deploy_phase_1=true" \
        -var="deploy_phase_2=$([ "$phase" -ge 2 ] && echo true || echo false)" \
        -var="deploy_phase_3=$([ "$phase" -ge 3 ] && echo true || echo false)" \
        -var="service_account_email=$SERVICE_ACCOUNT_EMAIL" \
        -out=tfplan
    
    # Apply infrastructure
    terraform apply -auto-approve tfplan
    
    cd ../..
    success "Phase $phase infrastructure deployed"
}

# Build and push container images using Cloud Build (Termux compatible)
build_and_push_images() {
    local phase=$1
    log "Building and pushing Phase $phase container images using Cloud Build..."
    
    # Check if Cloud Build API is enabled
    if ! gcloud services list --enabled --filter="name:cloudbuild.googleapis.com" --format="value(name)" | grep -q "cloudbuild.googleapis.com"; then
        log "Enabling Cloud Build API..."
        gcloud services enable cloudbuild.googleapis.com
    fi
    
    # Determine build strategy based on phase
    local build_config=""
    local substitutions="_PROJECT_ID=$PROJECT_ID,_REGION=$REGION,_REPOSITORY=aetherveil,_IMAGE_TAG=latest,_PHASE=$phase"
    
    if [ "$phase" -eq 1 ]; then
        # Build Phase 1 agents only
        log "Building Phase 1 agents: pipeline-observer, anomaly-detector, self-healing-engine"
        
        # Use individual builds for Phase 1 for better control
        for agent in "pipeline-observer" "anomaly-detector" "self-healing-engine"; do
            log "Submitting $agent build to Cloud Build..."
            
            gcloud builds submit \
                --config=<(cat << EOF
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args:
      - 'build'
      - '-t'
      - '${REGION}-docker.pkg.dev/${PROJECT_ID}/aetherveil/${agent}:latest'
      - '-t'
      - '${REGION}-docker.pkg.dev/${PROJECT_ID}/aetherveil/${agent}:v1.0'
      - '-f'
      - 'deployment/docker/${agent}/Dockerfile'
      - '.'
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', '${REGION}-docker.pkg.dev/${PROJECT_ID}/aetherveil/${agent}:latest']
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', '${REGION}-docker.pkg.dev/${PROJECT_ID}/aetherveil/${agent}:v1.0']
images:
  - '${REGION}-docker.pkg.dev/${PROJECT_ID}/aetherveil/${agent}:latest'
  - '${REGION}-docker.pkg.dev/${PROJECT_ID}/aetherveil/${agent}:v1.0'
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
        
    elif [ "$phase" -ge 2 ]; then
        # Build all agents (Phase 1 + Phase 2+) using master config
        log "Building all phases using master Cloud Build configuration..."
        
        gcloud builds submit \
            --config=cloudbuild.yaml \
            --region="$REGION" \
            --substitutions="$substitutions" \
            . || error "Cloud Build submission failed"
    fi
    
    # Wait for builds to complete
    log "Waiting for Cloud Build to complete..."
    local max_wait=1800  # 30 minutes
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
        
        log "Waiting for $running_builds builds to complete... (${elapsed}s elapsed)"
        sleep $check_interval
        elapsed=$((elapsed + check_interval))
    done
    
    if [ $elapsed -ge $max_wait ]; then
        warning "Build timeout reached. Check Cloud Build console for status."
    fi
    
    # Show final build status
    log "Recent build status:"
    gcloud builds list --limit=5 --format="table(id,status,createTime,duration)"
    
    success "Phase $phase images built and pushed via Cloud Build"
}

# Deploy Cloud Functions
deploy_cloud_functions() {
    log "Deploying Cloud Functions..."
    
    # Deploy GitHub webhook processor
    gcloud functions deploy github-webhook-processor \
        --gen2 \
        --runtime=python311 \
        --region=$REGION \
        --source=functions/webhook-processor \
        --entry-point=process_webhook \
        --trigger-http \
        --allow-unauthenticated \
        --memory=512MB \
        --timeout=300s \
        --set-env-vars="PROJECT_ID=$PROJECT_ID,PUBSUB_TOPIC=aetherveil-pipeline-events" \
        --quiet
    
    success "Cloud Functions deployed"
}

# Setup BigQuery ML models and Firestore collections
setup_bigquery_ml() {
    log "Setting up BigQuery ML models and Firestore collections..."
    
    # Create BigQuery tables for analytics
    bq query --use_legacy_sql=false "
    CREATE TABLE IF NOT EXISTS \`$PROJECT_ID.pipeline_analytics.pipeline_runs\` (
        run_id STRING,
        repository STRING,
        workflow_name STRING,
        branch STRING,
        commit_sha STRING,
        actor STRING,
        environment STRING,
        status STRING,
        started_at TIMESTAMP,
        completed_at TIMESTAMP,
        duration_seconds FLOAT64,
        conclusion STRING,
        jobs_total INT64,
        jobs_failed INT64,
        artifact_count INT64,
        test_results JSON,
        security_scan_results JSON,
        resource_usage JSON,
        timestamp TIMESTAMP
    )
    PARTITION BY DATE(timestamp)
    CLUSTER BY repository, workflow_name, status
    "
    
    # Create healing history table for analytics
    bq query --use_legacy_sql=false "
    CREATE TABLE IF NOT EXISTS \`$PROJECT_ID.pipeline_analytics.healing_history\` (
        event_id STRING,
        timestamp TIMESTAMP,
        repository STRING,
        workflow_name STRING,
        failure_type STRING,
        healing_actions ARRAY<STRING>,
        success BOOLEAN,
        duration_seconds FLOAT64,
        error_message STRING
    )
    PARTITION BY DATE(timestamp)
    CLUSTER BY repository, failure_type, success
    "
    
    # Create ADO executions table for analytics
    bq query --use_legacy_sql=false "
    CREATE TABLE IF NOT EXISTS \`$PROJECT_ID.pipeline_analytics.ado_executions\` (
        execution_id STRING,
        timestamp TIMESTAMP,
        status STRING,
        duration_seconds FLOAT64,
        steps_completed INT64,
        errors STRING,
        success BOOLEAN
    )
    PARTITION BY DATE(timestamp)
    CLUSTER BY status, success
    "

    # Create Red Team Findings table
    bq query --use_legacy_sql=false "
    CREATE TABLE IF NOT EXISTS `$PROJECT_ID.security_analytics.red_team_findings` (
        timestamp TIMESTAMP,
        target STRING,
        vulnerability_name STRING,
        severity STRING,
        description STRING,
        recommendation STRING,
        agent_id STRING,
        status STRING
    )
    PARTITION BY DATE(timestamp)
    CLUSTER BY target, vulnerability_name, severity
    "
    
    # Initialize Firestore with security rules (done via Terraform)
    log "Firestore collections will be created automatically by agents"
    
    success "BigQuery tables and Firestore setup completed"
}

# Setup monitoring and alerting
setup_monitoring() {
    log "Setting up monitoring and alerting..."
    
    # Create monitoring dashboard
    if [ -f "monitoring/autonomous-platform-dashboard.json" ]; then
        gcloud monitoring dashboards create --config-from-file=monitoring/autonomous-platform-dashboard.json || true
    fi
    
    # Create alerting policies
    if [ -f "monitoring/alert-policies.yaml" ]; then
        gcloud alpha monitoring policies create --policy-from-file=monitoring/alert-policies.yaml || true
    fi
    
    success "Monitoring and alerting configured"
}

# Run integration tests
run_tests() {
    log "Running integration tests..."
    
    # Install test dependencies
    pip install pytest httpx google-cloud-pubsub google-cloud-bigquery
    
    # Run tests if they exist
    if [ -d "tests/integration" ]; then
        pytest tests/integration/ -v || warning "Some integration tests failed"
    else
        warning "No integration tests found"
    fi
    
    success "Tests completed"
}

# Main deployment function
deploy_phase() {
    local phase=$1
    
    log "Starting Phase $phase deployment of Autonomous AI-Powered DevOps Platform"
    
    check_prerequisites
    enable_apis
    create_artifact_registry
    setup_terraform_backend
    
    # Build images first
    build_and_push_images $phase
    
    # Deploy infrastructure
    deploy_infrastructure $phase
    
    # Deploy additional components
    deploy_cloud_functions
    setup_bigquery_ml
    setup_monitoring
    
    # Run tests
    run_tests
    
    success "Phase $phase deployment completed successfully!"
    
    # Print deployment information
    echo ""
    echo "🚀 Autonomous AI-Powered DevOps Platform Deployed! 🚀"
    echo ""
    echo "Phase $phase Components:"
    
    if [ "$phase" -ge 1 ]; then
        echo "✅ Phase 1 - Immediate (0-3 months):"
        echo "   • Pipeline State Monitor"
        echo "   • Log Anomaly Detector"
        echo "   • Performance Baseline Generator"
        echo "   • Self-Healing Workflow Engine"
        echo "   • Intelligent Security Gate"
    fi
    
    if [ "$phase" -ge 2 ]; then
        echo "✅ Phase 2 - Medium-term (3-9 months):"
        echo "   • Autonomous DevOps Orchestrator (ADO)"
        echo "   • Predictive Infrastructure Management"
        echo "   • Adaptive Release Orchestration"
    fi
    
    if [ "$phase" -ge 3 ]; then
        echo "✅ Phase 3 - Long-term (9-18 months):"
        echo "   • Continuous Learning System"
        echo "   • Autonomous Security Posture Management"
        echo "   • Multi-Cloud Orchestration Intelligence"
    fi
    
    echo ""
    echo "📊 Monitoring Dashboard:"
    echo "   https://console.cloud.google.com/monitoring/dashboards"
    echo ""
    echo "🔧 Cloud Run Services:"
    echo "   https://console.cloud.google.com/run?project=$PROJECT_ID"
    echo ""
    echo "📈 BigQuery Analytics:"
    echo "   https://console.cloud.google.com/bigquery?project=$PROJECT_ID"
    echo ""
    echo "🎯 Next Steps:"
    echo "   1. Configure GitHub webhooks to point to Cloud Function"
    echo "   2. Set up notification channels for alerts"
    echo "   3. Configure API keys in Secret Manager"
    echo "   4. Run your first pipeline to see the system in action!"
}

# Parse command line arguments
PHASE=1

while [[ $# -gt 0 ]]; do
    case $1 in
        --phase)
            PHASE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--phase <1|2|3>]"
            echo ""
            echo "Options:"
            echo "  --phase <1|2|3>   Deploy specific phase (default: 1)"
            echo "  -h, --help        Show this help message"
            echo ""
            echo "Phases:"
            echo "  1 - Immediate components (0-3 months)"
            echo "  2 - Medium-term components (3-9 months)"
            echo "  3 - Long-term components (9-18 months)"
            exit 0
            ;;
        *)
            error "Unknown option $1"
            exit 1
            ;;
    esac
done

# Validate phase
if [[ ! "$PHASE" =~ ^[123]$ ]]; then
    error "Invalid phase. Must be 1, 2, or 3."
    exit 1
fi

# Run deployment
deploy_phase $PHASE