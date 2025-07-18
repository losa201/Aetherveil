#!/bin/bash

# Aetherveil Sentinel Kubernetes Deployment Script
# This script deploys the Aetherveil Sentinel swarm system to Kubernetes

set -euo pipefail

# Configuration
NAMESPACE="aetherveil-sentinel"
CHART_PATH="./helm/aetherveil-sentinel"
RELEASE_NAME="aetherveil-sentinel"
VALUES_FILE="values.yaml"
ENVIRONMENT="production"
TIMEOUT="600s"

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

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Help function
show_help() {
    cat << EOF
Usage: $0 [OPTIONS] [COMMAND]

Aetherveil Sentinel Kubernetes Deployment Script

Commands:
    install     Install Aetherveil Sentinel
    upgrade     Upgrade existing installation
    uninstall   Uninstall Aetherveil Sentinel
    status      Show deployment status
    logs        Show logs from components
    shell       Open shell in coordinator pod
    port-forward Port forward to coordinator
    test        Run deployment tests

Options:
    -n, --namespace NAMESPACE    Kubernetes namespace (default: aetherveil-sentinel)
    -e, --environment ENV        Environment (dev/staging/production) (default: production)
    -f, --values-file FILE       Values file path (default: values.yaml)
    -t, --timeout TIMEOUT       Timeout for operations (default: 600s)
    -h, --help                  Show this help message

Examples:
    $0 install                              # Install with default settings
    $0 install -n my-namespace -e dev       # Install in custom namespace for dev
    $0 upgrade -f custom-values.yaml       # Upgrade with custom values
    $0 uninstall                           # Uninstall the deployment
    $0 status                              # Show deployment status
    $0 logs coordinator                    # Show coordinator logs
    $0 port-forward                        # Port forward to coordinator UI

EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -n|--namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -f|--values-file)
                VALUES_FILE="$2"
                shift 2
                ;;
            -t|--timeout)
                TIMEOUT="$2"
                shift 2
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            install|upgrade|uninstall|status|logs|shell|port-forward|test)
                COMMAND="$1"
                shift
                ;;
            *)
                if [[ -z "${COMMAND:-}" ]]; then
                    log_error "Unknown option: $1"
                    show_help
                    exit 1
                else
                    COMPONENT="$1"
                    shift
                fi
                ;;
        esac
    done
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if kubectl is installed
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed. Please install kubectl first."
        exit 1
    fi
    
    # Check if helm is installed
    if ! command -v helm &> /dev/null; then
        log_error "helm is not installed. Please install helm first."
        exit 1
    fi
    
    # Check if cluster is accessible
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster. Please check your kubeconfig."
        exit 1
    fi
    
    # Check if chart exists
    if [[ ! -d "$CHART_PATH" ]]; then
        log_error "Helm chart not found at $CHART_PATH"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Create namespace if it doesn't exist
create_namespace() {
    log_info "Creating namespace $NAMESPACE..."
    
    if kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_warn "Namespace $NAMESPACE already exists"
    else
        kubectl create namespace "$NAMESPACE"
        kubectl label namespace "$NAMESPACE" name="$NAMESPACE"
        log_success "Namespace $NAMESPACE created"
    fi
}

# Setup RBAC
setup_rbac() {
    log_info "Setting up RBAC..."
    
    # Apply cluster-wide RBAC if needed
    if [[ -f "rbac-cluster.yaml" ]]; then
        kubectl apply -f rbac-cluster.yaml
        log_success "Cluster RBAC applied"
    fi
}

# Install cert-manager if not present
install_cert_manager() {
    log_info "Checking cert-manager installation..."
    
    if ! kubectl get namespace cert-manager &> /dev/null; then
        log_info "Installing cert-manager..."
        kubectl apply -f https://github.com/jetstack/cert-manager/releases/download/v1.13.2/cert-manager.yaml
        
        # Wait for cert-manager to be ready
        log_info "Waiting for cert-manager to be ready..."
        kubectl wait --for=condition=ready pod -l app=cert-manager -n cert-manager --timeout=300s
        kubectl wait --for=condition=ready pod -l app=cainjector -n cert-manager --timeout=300s
        kubectl wait --for=condition=ready pod -l app=webhook -n cert-manager --timeout=300s
        
        log_success "cert-manager installed successfully"
    else
        log_success "cert-manager already installed"
    fi
}

# Install or upgrade Aetherveil Sentinel
install_aetherveil() {
    log_info "Installing Aetherveil Sentinel..."
    
    # Update helm dependencies
    log_info "Updating helm dependencies..."
    helm dependency update "$CHART_PATH"
    
    # Install/upgrade the chart
    helm upgrade --install "$RELEASE_NAME" "$CHART_PATH" \
        --namespace "$NAMESPACE" \
        --values "$VALUES_FILE" \
        --timeout "$TIMEOUT" \
        --wait \
        --wait-for-jobs \
        --create-namespace
    
    log_success "Aetherveil Sentinel installed successfully"
}

# Upgrade Aetherveil Sentinel
upgrade_aetherveil() {
    log_info "Upgrading Aetherveil Sentinel..."
    
    # Update helm dependencies
    log_info "Updating helm dependencies..."
    helm dependency update "$CHART_PATH"
    
    # Upgrade the chart
    helm upgrade "$RELEASE_NAME" "$CHART_PATH" \
        --namespace "$NAMESPACE" \
        --values "$VALUES_FILE" \
        --timeout "$TIMEOUT" \
        --wait \
        --wait-for-jobs \
        --reuse-values
    
    log_success "Aetherveil Sentinel upgraded successfully"
}

# Uninstall Aetherveil Sentinel
uninstall_aetherveil() {
    log_warn "Uninstalling Aetherveil Sentinel..."
    read -p "Are you sure you want to uninstall Aetherveil Sentinel? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        helm uninstall "$RELEASE_NAME" --namespace "$NAMESPACE"
        
        # Optionally delete namespace
        read -p "Delete namespace $NAMESPACE? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            kubectl delete namespace "$NAMESPACE"
            log_success "Namespace $NAMESPACE deleted"
        fi
        
        log_success "Aetherveil Sentinel uninstalled successfully"
    else
        log_info "Uninstall cancelled"
    fi
}

# Show deployment status
show_status() {
    log_info "Showing deployment status..."
    
    echo
    echo "=== Helm Release Status ==="
    helm status "$RELEASE_NAME" --namespace "$NAMESPACE" || true
    
    echo
    echo "=== Pods Status ==="
    kubectl get pods -n "$NAMESPACE" -o wide
    
    echo
    echo "=== Services Status ==="
    kubectl get services -n "$NAMESPACE" -o wide
    
    echo
    echo "=== Ingress Status ==="
    kubectl get ingress -n "$NAMESPACE" -o wide
    
    echo
    echo "=== PVC Status ==="
    kubectl get pvc -n "$NAMESPACE" -o wide
    
    echo
    echo "=== HPA Status ==="
    kubectl get hpa -n "$NAMESPACE" -o wide
    
    echo
    echo "=== Events ==="
    kubectl get events -n "$NAMESPACE" --sort-by=.metadata.creationTimestamp
}

# Show logs
show_logs() {
    local component="${COMPONENT:-coordinator}"
    
    log_info "Showing logs for $component..."
    
    case $component in
        coordinator)
            kubectl logs -n "$NAMESPACE" -l app.kubernetes.io/component=coordinator --tail=100 -f
            ;;
        agent|agents)
            kubectl logs -n "$NAMESPACE" -l app.kubernetes.io/component=agent --tail=100 -f
            ;;
        reconnaissance)
            kubectl logs -n "$NAMESPACE" -l app.kubernetes.io/agent-type=reconnaissance --tail=100 -f
            ;;
        scanner)
            kubectl logs -n "$NAMESPACE" -l app.kubernetes.io/agent-type=scanner --tail=100 -f
            ;;
        exploiter)
            kubectl logs -n "$NAMESPACE" -l app.kubernetes.io/agent-type=exploiter --tail=100 -f
            ;;
        osint)
            kubectl logs -n "$NAMESPACE" -l app.kubernetes.io/agent-type=osint --tail=100 -f
            ;;
        stealth)
            kubectl logs -n "$NAMESPACE" -l app.kubernetes.io/agent-type=stealth --tail=100 -f
            ;;
        *)
            log_error "Unknown component: $component"
            log_info "Available components: coordinator, agent, reconnaissance, scanner, exploiter, osint, stealth"
            exit 1
            ;;
    esac
}

# Open shell in coordinator pod
open_shell() {
    log_info "Opening shell in coordinator pod..."
    
    POD_NAME=$(kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/component=coordinator -o jsonpath='{.items[0].metadata.name}')
    
    if [[ -n "$POD_NAME" ]]; then
        kubectl exec -it "$POD_NAME" -n "$NAMESPACE" -- /bin/bash
    else
        log_error "No coordinator pod found"
        exit 1
    fi
}

# Port forward to coordinator
port_forward() {
    log_info "Setting up port forwarding to coordinator..."
    
    POD_NAME=$(kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/component=coordinator -o jsonpath='{.items[0].metadata.name}')
    
    if [[ -n "$POD_NAME" ]]; then
        log_info "Access coordinator at http://localhost:8080"
        kubectl port-forward "$POD_NAME" -n "$NAMESPACE" 8080:8080
    else
        log_error "No coordinator pod found"
        exit 1
    fi
}

# Run deployment tests
run_tests() {
    log_info "Running deployment tests..."
    
    # Test coordinator health
    log_info "Testing coordinator health..."
    kubectl run test-coordinator --image=curlimages/curl:latest --rm -i --tty --restart=Never --namespace="$NAMESPACE" \
        -- curl -f "http://${RELEASE_NAME}-coordinator:8080/health" || log_error "Coordinator health check failed"
    
    # Test agent connectivity
    log_info "Testing agent connectivity..."
    kubectl run test-agent --image=curlimages/curl:latest --rm -i --tty --restart=Never --namespace="$NAMESPACE" \
        -- curl -f "http://${RELEASE_NAME}-agent-reconnaissance:8080/health" || log_error "Agent health check failed"
    
    # Test Redis connectivity
    log_info "Testing Redis connectivity..."
    kubectl run test-redis --image=redis:latest --rm -i --tty --restart=Never --namespace="$NAMESPACE" \
        -- redis-cli -h "${RELEASE_NAME}-redis-master" ping || log_error "Redis connectivity test failed"
    
    # Test PostgreSQL connectivity
    log_info "Testing PostgreSQL connectivity..."
    kubectl run test-postgres --image=postgres:latest --rm -i --tty --restart=Never --namespace="$NAMESPACE" \
        --env="PGPASSWORD=aetherveil-postgres-password" \
        -- psql -h "${RELEASE_NAME}-postgresql" -U aetherveil -d aetherveil -c "SELECT 1;" || log_error "PostgreSQL connectivity test failed"
    
    log_success "All tests passed"
}

# Main execution
main() {
    parse_args "$@"
    
    if [[ -z "${COMMAND:-}" ]]; then
        show_help
        exit 1
    fi
    
    case $COMMAND in
        install)
            check_prerequisites
            create_namespace
            setup_rbac
            install_cert_manager
            install_aetherveil
            ;;
        upgrade)
            check_prerequisites
            upgrade_aetherveil
            ;;
        uninstall)
            uninstall_aetherveil
            ;;
        status)
            show_status
            ;;
        logs)
            show_logs
            ;;
        shell)
            open_shell
            ;;
        port-forward)
            port_forward
            ;;
        test)
            run_tests
            ;;
        *)
            log_error "Unknown command: $COMMAND"
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"