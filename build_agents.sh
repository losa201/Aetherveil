#!/bin/bash

# Aetherveil Autonomous AI DevOps Platform - Cloud-Native Build Script
# Optimized for Termux Ubuntu (proot-distro) environments without Docker
# Uses GCP Cloud Build for all container image builds

set -e

# Configuration
PROJECT_ID="${PROJECT_ID:-tidy-computing-465909-i3}"
REGION="${REGION:-us-central1}"
REPOSITORY="aetherveil"
IMAGE_TAG="${IMAGE_TAG:-latest}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

# Agent configurations for each phase
declare -A PHASE_1_AGENTS=(
    ["pipeline-observer"]="agents/pipeline_observer"
    ["anomaly-detector"]="agents/anomaly_detector"
    ["self-healing-engine"]="agents/self_healing_engine"
)

declare -A PHASE_2_AGENTS=(
    ["ado-orchestrator"]="agents/ado_orchestrator"
    ["red-team-pentester-v2"]="agents/red_team_pentester_v2"
)

declare -A PHASE_3_AGENTS=(
    # Phase 3 agents will be added when implemented
    # ["continuous-learning"]="agents/continuous_learning"
    # ["security-posture"]="agents/security_posture"
    # ["multi-cloud-orchestrator"]="agents/multi_cloud_orchestrator"
)

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites for cloud-native build..."
    
    # Check if gcloud is installed and authenticated
    if ! command -v gcloud &> /dev/null; then
        error "gcloud CLI is not installed. Please install it first."
        exit 1
    fi
    
    # Check if authenticated to GCP
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q "@"; then
        error "Please authenticate to GCP first: gcloud auth login"
        exit 1
    fi
    
    # Verify project ID
    current_project=$(gcloud config get-value project 2>/dev/null)
    if [ "$current_project" != "$PROJECT_ID" ]; then
        log "Setting project to $PROJECT_ID..."
        gcloud config set project $PROJECT_ID
    fi
    
    # Check if Cloud Build API is enabled
    if ! gcloud services list --enabled --filter="name:cloudbuild.googleapis.com" --format="value(name)" | grep -q "cloudbuild.googleapis.com"; then
        log "Enabling Cloud Build API..."
        gcloud services enable cloudbuild.googleapis.com
    fi
    
    # Check if Artifact Registry repository exists
    if ! gcloud artifacts repositories describe $REPOSITORY --location=$REGION &>/dev/null; then
        log "Creating Artifact Registry repository..."
        gcloud artifacts repositories create $REPOSITORY \
            --repository-format=docker \
            --location=$REGION \
            --description="Aetherveil Autonomous DevOps Platform"
    fi
    
    success "Prerequisites check completed"
}

# Generate Cloud Build configuration for an agent
generate_cloudbuild_config() {
    local agent_name=$1
    local agent_path=$2
    local dockerfile_path="deployment/docker/$agent_name/Dockerfile"
    
    cat > "cloudbuild-${agent_name}.yaml" << EOF
steps:
  # Build the container image
  - name: 'gcr.io/cloud-builders/docker'
    args:
      - 'build'
      - '-t'
      - '${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${agent_name}:${IMAGE_TAG}'
      - '-t'
      - '${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${agent_name}:latest'
      - '-f'
      - '${dockerfile_path}'
      - '.'

  # Push the container image to Artifact Registry
  - name: 'gcr.io/cloud-builders/docker'
    args:
      - 'push'
      - '${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${agent_name}:${IMAGE_TAG}'

  - name: 'gcr.io/cloud-builders/docker'
    args:
      - 'push'
      - '${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${agent_name}:latest'

# Store images in Artifact Registry
images:
  - '${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${agent_name}:${IMAGE_TAG}'
  - '${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${agent_name}:latest'

# Build timeout (20 minutes)
timeout: '1200s'

options:
  # Use high-performance machine type for faster builds
  machineType: 'E2_HIGHCPU_8'
  # Enable concurrent builds
  substitutionOption: 'ALLOW_LOOSE'
  # Enable build logs
  logging: CLOUD_LOGGING_ONLY

substitutions:
  _AGENT_NAME: '${agent_name}'
  _AGENT_PATH: '${agent_path}'
  _PROJECT_ID: '${PROJECT_ID}'
  _REGION: '${REGION}'
  _REPOSITORY: '${REPOSITORY}'
  _IMAGE_TAG: '${IMAGE_TAG}'
EOF
}

# Build a single agent using Cloud Build
build_agent() {
    local agent_name=$1
    local agent_path=$2
    
    log "Building agent: $agent_name"
    
    # Generate Cloud Build configuration
    generate_cloudbuild_config "$agent_name" "$agent_path"
    
    # Submit build to Cloud Build
    log "Submitting $agent_name build to Cloud Build..."
    
    gcloud builds submit \
        --config="cloudbuild-${agent_name}.yaml" \
        --region="$REGION" \
        --async \
        --substitutions="_AGENT_NAME=${agent_name},_AGENT_PATH=${agent_path}" \
        .
    
    success "Build submitted for $agent_name"
}

# Build all agents for a specific phase
build_phase_agents() {
    local phase=$1
    
    case $phase in
        1)
            log "Building Phase 1 agents..."
            for agent_name in "${!PHASE_1_AGENTS[@]}"; do
                build_agent "$agent_name" "${PHASE_1_AGENTS[$agent_name]}"
            done
            ;;
        2)
            log "Building Phase 2 agents..."
            for agent_name in "${!PHASE_2_AGENTS[@]}"; do
                build_agent "$agent_name" "${PHASE_2_AGENTS[$agent_name]}"
            done
            ;;
        3)
            log "Building Phase 3 agents..."
            for agent_name in "${!PHASE_3_AGENTS[@]}"; do
                build_agent "$agent_name" "${PHASE_3_AGENTS[$agent_name]}"
            done
            ;;
        *)
            error "Invalid phase: $phase. Must be 1, 2, or 3."
            exit 1
            ;;
    esac
}

# Wait for all builds to complete
wait_for_builds() {
    log "Waiting for all builds to complete..."
    
    local max_wait=1800  # 30 minutes
    local elapsed=0
    local check_interval=30
    
    while [ $elapsed -lt $max_wait ]; do
        # Check for running builds
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
        warning "Build timeout reached. Some builds may still be running."
    fi
    
    # Show final build status
    log "Final build status:"
    gcloud builds list --limit=10 --format="table(id,status,createTime,duration)"
}

# Build all agents across phases
build_all_phases() {
    log "Building all phases of Aetherveil Autonomous AI DevOps Platform"
    
    # Build Phase 1 agents (immediate deployment)
    build_phase_agents 1
    
    # Build Phase 2 agents (medium-term)
    build_phase_agents 2
    
    # Phase 3 agents will be built when implemented
    # build_phase_agents 3
    
    # Wait for all builds to complete
    wait_for_builds
    
    # Cleanup temporary files
    rm -f cloudbuild-*.yaml
    
    success "All agent builds completed!"
}

# Clean up previous builds
cleanup_builds() {
    log "Cleaning up previous Cloud Build configurations..."
    rm -f cloudbuild-*.yaml
    success "Cleanup completed"
}

# Show build status
show_build_status() {
    log "Current Cloud Build status:"
    gcloud builds list --limit=20 --format="table(id,status,source.repoSource.repoName,createTime,duration)"
}

# Show available images in Artifact Registry
show_images() {
    log "Available images in Artifact Registry:"
    gcloud artifacts docker images list \
        "${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}" \
        --format="table(IMAGE,TAGS,CREATE_TIME,UPDATE_TIME)" \
        --sort-by="~UPDATE_TIME"
}

# Print usage information
usage() {
    echo "Usage: $0 [OPTIONS] COMMAND"
    echo ""
    echo "Commands:"
    echo "  phase <1|2|3>     Build agents for specific phase"
    echo "  all               Build all phases"
    echo "  status            Show current build status"
    echo "  images            Show available images in registry"
    echo "  cleanup           Clean up temporary files"
    echo "  wait              Wait for running builds to complete"
    echo ""
    echo "Options:"
    echo "  --project-id      GCP Project ID (default: $PROJECT_ID)"
    echo "  --region          GCP Region (default: $REGION)"
    echo "  --tag             Image tag (default: $IMAGE_TAG)"
    echo "  -h, --help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 phase 1        # Build Phase 1 agents only"
    echo "  $0 all            # Build all phases"
    echo "  $0 status         # Check build status"
    echo "  $0 images         # List built images"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --project-id)
            PROJECT_ID="$2"
            shift 2
            ;;
        --region)
            REGION="$2"
            shift 2
            ;;
        --tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        phase)
            COMMAND="phase"
            PHASE="$2"
            shift 2
            ;;
        all)
            COMMAND="all"
            shift
            ;;
        status)
            COMMAND="status"
            shift
            ;;
        images)
            COMMAND="images"
            shift
            ;;
        cleanup)
            COMMAND="cleanup"
            shift
            ;;
        wait)
            COMMAND="wait"
            shift
            ;;
        *)
            error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Execute command
case "${COMMAND:-all}" in
    phase)
        if [ -z "$PHASE" ]; then
            error "Phase number required. Use: $0 phase <1|2|3>"
            exit 1
        fi
        check_prerequisites
        build_phase_agents "$PHASE"
        wait_for_builds
        ;;
    all)
        check_prerequisites
        build_all_phases
        ;;
    status)
        show_build_status
        ;;
    images)
        show_images
        ;;
    cleanup)
        cleanup_builds
        ;;
    wait)
        wait_for_builds
        ;;
    *)
        error "Invalid command. Use -h for help."
        exit 1
        ;;
esac

success "Build script execution completed!"