#!/bin/bash

# Aetherveil Deployment Verification Script
# Validates that all components are properly deployed and functional

set -e

PROJECT_ID="${PROJECT_ID:-tidy-computing-465909-i3}"
REGION="${REGION:-us-central1}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

success() { echo -e "${GREEN}[✅]${NC} $1"; }
error() { echo -e "${RED}[❌]${NC} $1"; }
warning() { echo -e "${YELLOW}[⚠️]${NC} $1"; }
info() { echo -e "${BLUE}[ℹ️]${NC} $1"; }

echo "🔍 Verifying Aetherveil Autonomous AI DevOps Platform Deployment"
echo "Project: $PROJECT_ID | Region: $REGION"
echo "============================================================="

# Check GCP authentication
info "Checking GCP authentication..."
if gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q "@"; then
    success "GCP authentication verified"
else
    error "GCP authentication failed. Run: gcloud auth login"
    exit 1
fi

# Check project setting
current_project=$(gcloud config get-value project 2>/dev/null)
if [ "$current_project" = "$PROJECT_ID" ]; then
    success "Project correctly set to $PROJECT_ID"
else
    warning "Project not set correctly. Current: $current_project"
    gcloud config set project $PROJECT_ID
fi

# Check required APIs
info "Checking required APIs..."
required_apis=(
    "cloudbuild.googleapis.com"
    "run.googleapis.com"
    "artifactregistry.googleapis.com"
    "firestore.googleapis.com"
    "bigquery.googleapis.com"
    "pubsub.googleapis.com"
    "cloudfunctions.googleapis.com"
)

for api in "${required_apis[@]}"; do
    if gcloud services list --enabled --filter="name:$api" --format="value(name)" | grep -q "$api"; then
        success "$api enabled"
    else
        warning "$api not enabled"
    fi
done

# Check Artifact Registry
info "Checking Artifact Registry..."
if gcloud artifacts repositories describe aetherveil --location=$REGION &>/dev/null; then
    success "Artifact Registry repository 'aetherveil' exists"
    
    # Check for images
    image_count=$(gcloud artifacts docker images list ${REGION}-docker.pkg.dev/${PROJECT_ID}/aetherveil --format="value(IMAGE)" | wc -l)
    if [ "$image_count" -gt 0 ]; then
        success "$image_count container images found in registry"
        gcloud artifacts docker images list ${REGION}-docker.pkg.dev/${PROJECT_ID}/aetherveil --format="table(IMAGE,TAGS,CREATE_TIME)" --limit=5
    else
        warning "No container images found. Run: ./build_agents.sh all"
    fi
else
    error "Artifact Registry repository 'aetherveil' not found"
fi

# Check Cloud Run services
info "Checking Cloud Run services..."
run_services=$(gcloud run services list --region=$REGION --format="value(metadata.name)" | grep -E "(pipeline-observer|anomaly-detector|self-healing-engine|ado-orchestrator)" | wc -l)

if [ "$run_services" -gt 0 ]; then
    success "$run_services Cloud Run services deployed"
    gcloud run services list --region=$REGION --format="table(metadata.name,status.url,status.conditions[0].type)"
else
    warning "No Aetherveil Cloud Run services found. Check deployment status."
fi

# Check Firestore
info "Checking Firestore database..."
if gcloud firestore databases describe --database="(default)" &>/dev/null; then
    success "Firestore database configured"
else
    warning "Firestore database not found or not accessible"
fi

# Check BigQuery
info "Checking BigQuery datasets..."
if bq ls | grep -q "pipeline_analytics"; then
    success "BigQuery dataset 'pipeline_analytics' exists"
    
    # Check tables
    table_count=$(bq ls pipeline_analytics | grep -c "TABLE" || echo "0")
    if [ "$table_count" -gt 0 ]; then
        success "$table_count tables found in pipeline_analytics dataset"
    else
        warning "No tables found in pipeline_analytics dataset"
    fi
else
    warning "BigQuery dataset 'pipeline_analytics' not found"
fi

# Check Pub/Sub topics
info "Checking Pub/Sub topics..."
pubsub_topics=$(gcloud pubsub topics list --format="value(name)" | grep -E "(pipeline-events|anomaly-alerts|healing-actions)" | wc -l)

if [ "$pubsub_topics" -gt 0 ]; then
    success "$pubsub_topics Pub/Sub topics configured"
else
    warning "Aetherveil Pub/Sub topics not found"
fi

# Check Cloud Functions
info "Checking Cloud Functions..."
if gcloud functions list --format="value(name)" | grep -q "github-webhook-processor"; then
    success "GitHub webhook processor function deployed"
    
    # Get function URL
    function_url=$(gcloud functions describe github-webhook-processor --region=$REGION --format="value(httpsTrigger.url)")
    info "Webhook URL: $function_url"
else
    warning "GitHub webhook processor function not found"
fi

# Check Secret Manager
info "Checking Secret Manager..."
secret_count=$(gcloud secrets list --format="value(name)" | grep -E "(github-token|openai-api-key)" | wc -l)

if [ "$secret_count" -gt 0 ]; then
    success "$secret_count secrets configured in Secret Manager"
else
    warning "No API secrets found. Configure secrets for full functionality."
fi

# Check Cloud Build recent activity
info "Checking recent Cloud Build activity..."
recent_builds=$(gcloud builds list --limit=5 --format="value(id)" | wc -l)

if [ "$recent_builds" -gt 0 ]; then
    success "$recent_builds recent builds found"
    echo "Recent builds:"
    gcloud builds list --limit=5 --format="table(id,status,createTime,duration)"
else
    info "No recent builds found"
fi

# Health check for deployed services
info "Performing health checks..."
run_services_list=$(gcloud run services list --region=$REGION --format="value(metadata.name,status.url)" | grep -E "(pipeline-observer|anomaly-detector|self-healing-engine|ado-orchestrator)")

if [ -n "$run_services_list" ]; then
    while IFS=$'\t' read -r service_name service_url; do
        if [ -n "$service_url" ]; then
            info "Testing health check for $service_name..."
            if curl -s --max-time 10 "$service_url/health" | grep -q "healthy\|ok\|status"; then
                success "$service_name is responding to health checks"
            else
                warning "$service_name health check failed or not implemented"
            fi
        fi
    done <<< "$run_services_list"
fi

# Summary
echo ""
echo "============================================================="
echo "🎯 DEPLOYMENT VERIFICATION SUMMARY"
echo "============================================================="

# Count successful components
components_ok=0
total_components=8

# Check each component
if gcloud artifacts repositories describe aetherveil --location=$REGION &>/dev/null; then
    ((components_ok++))
fi

if [ "$run_services" -gt 0 ]; then
    ((components_ok++))
fi

if gcloud firestore databases describe --database="(default)" &>/dev/null; then
    ((components_ok++))
fi

if bq ls | grep -q "pipeline_analytics"; then
    ((components_ok++))
fi

if [ "$pubsub_topics" -gt 0 ]; then
    ((components_ok++))
fi

if gcloud functions list --format="value(name)" | grep -q "github-webhook-processor"; then
    ((components_ok++))
fi

if [ "$image_count" -gt 0 ]; then
    ((components_ok++))
fi

# Add one more for overall API enablement
if [ "${#required_apis[@]}" -eq "$(gcloud services list --enabled --filter="name:($(IFS='|'; echo "${required_apis[*]}"))" --format="value(name)" | wc -l)" ]; then
    ((components_ok++))
fi

success "$components_ok/$total_components core components verified"

if [ "$components_ok" -eq "$total_components" ]; then
    echo ""
    success "🎉 DEPLOYMENT VERIFICATION PASSED!"
    success "Aetherveil Autonomous AI DevOps Platform is fully operational"
    echo ""
    info "Next steps:"
    echo "  1. Configure GitHub webhooks to point to your Cloud Function"
    echo "  2. Add API keys to Secret Manager (OpenAI, GitHub tokens)"
    echo "  3. Run your first pipeline to test autonomous agents"
    echo "  4. Monitor the platform via Cloud Console dashboards"
elif [ "$components_ok" -gt 4 ]; then
    echo ""
    warning "⚠️  DEPLOYMENT PARTIALLY COMPLETE"
    warning "Some components need attention. Check warnings above."
    echo ""
    info "To complete deployment:"
    echo "  1. Run: ./deploy.sh --phase <1|2|3>"
    echo "  2. Run: ./build_agents.sh all"
    echo "  3. Check Terraform deployment: cd infra/terraform && terraform plan"
else
    echo ""
    error "❌ DEPLOYMENT VERIFICATION FAILED"
    error "Multiple components missing. Please check deployment."
    echo ""
    info "To start deployment:"
    echo "  1. Run: ./deploy.sh --phase 1"
    echo "  2. Monitor progress with: ./verify_deployment.sh"
fi

echo ""
echo "============================================================="