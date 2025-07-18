#!/bin/bash

# Aetherveil Sentinel Deployment Script
# Comprehensive deployment with Docker Swarm and Kubernetes support

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEPLOYMENT_MODE="${1:-docker-compose}"
ENVIRONMENT="${2:-development}"
SCALING_PROFILE="${3:-default}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running. Please start Docker first."
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Setup directories
setup_directories() {
    log_info "Setting up directories..."
    
    mkdir -p "${PROJECT_DIR}/data"/{coordinator,agents/{recon-1,recon-2,scanner-1,exploiter-1,osint-1,stealth-1}}
    mkdir -p "${PROJECT_DIR}/logs"/{coordinator,agents/{recon-1,recon-2,scanner-1,exploiter-1,osint-1,stealth-1}}
    mkdir -p "${PROJECT_DIR}/config"
    
    # Set permissions
    chmod -R 755 "${PROJECT_DIR}/data"
    chmod -R 755 "${PROJECT_DIR}/logs"
    
    log_success "Directories setup completed"
}

# Build Docker images
build_images() {
    log_info "Building Docker images..."
    
    cd "${PROJECT_DIR}"
    
    # Build base image
    log_info "Building base agent image..."
    docker build -t aetherveil/base-agent:latest -f deployment/docker/base/Dockerfile .
    
    # Build specific agent images
    for agent_type in reconnaissance scanner exploiter osint stealth; do
        log_info "Building ${agent_type} agent image..."
        docker build -t aetherveil/${agent_type}-agent:latest -f deployment/docker/${agent_type}/Dockerfile .
    done
    
    # Build coordinator image
    log_info "Building coordinator image..."
    docker build -t aetherveil/coordinator:latest -f deployment/docker/coordinator/Dockerfile .
    
    log_success "Docker images built successfully"
}

# Deploy with Docker Compose
deploy_docker_compose() {
    log_info "Deploying with Docker Compose..."
    
    cd "${PROJECT_DIR}"
    
    # Create environment file
    cat > .env << EOF
# Aetherveil Sentinel Environment Configuration
ENVIRONMENT=${ENVIRONMENT}
COORDINATOR_HOST=coordinator
COORDINATOR_PORT=8000
ZMQ_PORT=5555
GRPC_PORT=50051
DATABASE_URL=sqlite:///app/data/aetherveil.db
REDIS_URL=redis://redis:6379/0
NEO4J_URI=bolt://neo4j:7687
NEO4J_AUTH=neo4j/aetherveil123
LOG_LEVEL=INFO
SCALING_PROFILE=${SCALING_PROFILE}
EOF
    
    # Deploy services
    docker-compose -f deployment/docker-compose.yml up -d
    
    log_success "Docker Compose deployment completed"
}

# Deploy with Docker Swarm
deploy_docker_swarm() {
    log_info "Deploying with Docker Swarm..."
    
    # Initialize swarm if not already initialized
    if ! docker info --format '{{.Swarm.LocalNodeState}}' | grep -q active; then
        log_info "Initializing Docker Swarm..."
        docker swarm init
    fi
    
    # Create overlay network
    docker network create --driver overlay aetherveil-swarm-network || true
    
    # Deploy stack
    docker stack deploy -c "${PROJECT_DIR}/deployment/docker-swarm.yml" aetherveil
    
    log_success "Docker Swarm deployment completed"
}

# Deploy with Kubernetes
deploy_kubernetes() {
    log_info "Deploying with Kubernetes..."
    
    # Check if kubectl is available
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed. Please install kubectl first."
        exit 1
    fi
    
    # Create namespace
    kubectl create namespace aetherveil-sentinel || true
    
    # Deploy services
    kubectl apply -f "${PROJECT_DIR}/deployment/kubernetes/" -n aetherveil-sentinel
    
    log_success "Kubernetes deployment completed"
}

# Scale agents based on profile
scale_agents() {
    log_info "Scaling agents based on profile: ${SCALING_PROFILE}"
    
    case "${SCALING_PROFILE}" in
        "minimal")
            RECON_REPLICAS=1
            SCANNER_REPLICAS=1
            EXPLOITER_REPLICAS=1
            OSINT_REPLICAS=1
            STEALTH_REPLICAS=1
            ;;
        "default")
            RECON_REPLICAS=2
            SCANNER_REPLICAS=2
            EXPLOITER_REPLICAS=1
            OSINT_REPLICAS=1
            STEALTH_REPLICAS=1
            ;;
        "performance")
            RECON_REPLICAS=4
            SCANNER_REPLICAS=4
            EXPLOITER_REPLICAS=2
            OSINT_REPLICAS=2
            STEALTH_REPLICAS=2
            ;;
        "enterprise")
            RECON_REPLICAS=8
            SCANNER_REPLICAS=6
            EXPLOITER_REPLICAS=4
            OSINT_REPLICAS=4
            STEALTH_REPLICAS=3
            ;;
        *)
            log_warning "Unknown scaling profile: ${SCALING_PROFILE}, using default"
            RECON_REPLICAS=2
            SCANNER_REPLICAS=2
            EXPLOITER_REPLICAS=1
            OSINT_REPLICAS=1
            STEALTH_REPLICAS=1
            ;;
    esac
    
    if [ "${DEPLOYMENT_MODE}" == "docker-swarm" ]; then
        docker service scale \
            aetherveil_reconnaissance=${RECON_REPLICAS} \
            aetherveil_scanner=${SCANNER_REPLICAS} \
            aetherveil_exploiter=${EXPLOITER_REPLICAS} \
            aetherveil_osint=${OSINT_REPLICAS} \
            aetherveil_stealth=${STEALTH_REPLICAS}
    elif [ "${DEPLOYMENT_MODE}" == "kubernetes" ]; then
        kubectl scale deployment reconnaissance-deployment --replicas=${RECON_REPLICAS} -n aetherveil-sentinel
        kubectl scale deployment scanner-deployment --replicas=${SCANNER_REPLICAS} -n aetherveil-sentinel
        kubectl scale deployment exploiter-deployment --replicas=${EXPLOITER_REPLICAS} -n aetherveil-sentinel
        kubectl scale deployment osint-deployment --replicas=${OSINT_REPLICAS} -n aetherveil-sentinel
        kubectl scale deployment stealth-deployment --replicas=${STEALTH_REPLICAS} -n aetherveil-sentinel
    fi
    
    log_success "Agent scaling completed"
}

# Health check
health_check() {
    log_info "Performing health check..."
    
    # Wait for services to start
    sleep 30
    
    # Check coordinator health
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        log_success "Coordinator is healthy"
    else
        log_error "Coordinator health check failed"
        return 1
    fi
    
    # Check Redis
    if docker exec aetherveil-redis redis-cli ping > /dev/null 2>&1; then
        log_success "Redis is healthy"
    else
        log_error "Redis health check failed"
        return 1
    fi
    
    # Check Neo4j
    if docker exec aetherveil-neo4j cypher-shell -u neo4j -p aetherveil123 "RETURN 1" > /dev/null 2>&1; then
        log_success "Neo4j is healthy"
    else
        log_error "Neo4j health check failed"
        return 1
    fi
    
    log_success "Health check completed successfully"
}

# Cleanup function
cleanup() {
    log_info "Cleaning up deployment..."
    
    case "${DEPLOYMENT_MODE}" in
        "docker-compose")
            docker-compose -f "${PROJECT_DIR}/deployment/docker-compose.yml" down -v
            ;;
        "docker-swarm")
            docker stack rm aetherveil
            docker network rm aetherveil-swarm-network || true
            ;;
        "kubernetes")
            kubectl delete namespace aetherveil-sentinel
            ;;
    esac
    
    log_success "Cleanup completed"
}

# Show deployment info
show_deployment_info() {
    log_info "Deployment Information:"
    echo "===================="
    echo "Mode: ${DEPLOYMENT_MODE}"
    echo "Environment: ${ENVIRONMENT}"
    echo "Scaling Profile: ${SCALING_PROFILE}"
    echo "Project Directory: ${PROJECT_DIR}"
    echo ""
    echo "Services:"
    echo "- Coordinator: http://localhost:8000"
    echo "- Neo4j Browser: http://localhost:7474"
    echo "- Redis: localhost:6379"
    echo ""
    echo "Useful Commands:"
    echo "- View logs: docker-compose logs -f [service]"
    echo "- Check status: docker-compose ps"
    echo "- Scale service: docker-compose up -d --scale [service]=[replicas]"
    echo "- Stop deployment: docker-compose down"
    echo "- Cleanup: docker-compose down -v"
    echo ""
}

# Main deployment flow
main() {
    log_info "Starting Aetherveil Sentinel deployment..."
    log_info "Mode: ${DEPLOYMENT_MODE}, Environment: ${ENVIRONMENT}, Scaling: ${SCALING_PROFILE}"
    
    # Handle cleanup flag
    if [ "$1" == "cleanup" ]; then
        cleanup
        exit 0
    fi
    
    # Check prerequisites
    check_prerequisites
    
    # Setup directories
    setup_directories
    
    # Build images
    build_images
    
    # Deploy based on mode
    case "${DEPLOYMENT_MODE}" in
        "docker-compose")
            deploy_docker_compose
            ;;
        "docker-swarm")
            deploy_docker_swarm
            ;;
        "kubernetes")
            deploy_kubernetes
            ;;
        *)
            log_error "Unknown deployment mode: ${DEPLOYMENT_MODE}"
            log_info "Supported modes: docker-compose, docker-swarm, kubernetes"
            exit 1
            ;;
    esac
    
    # Scale agents
    scale_agents
    
    # Health check
    if health_check; then
        show_deployment_info
        log_success "Aetherveil Sentinel deployment completed successfully!"
    else
        log_error "Deployment health check failed"
        exit 1
    fi
}

# Trap cleanup on exit
trap cleanup EXIT

# Run main function
main "$@"