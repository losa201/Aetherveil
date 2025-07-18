# Aetherveil Sentinel Kubernetes Deployment

This directory contains the Kubernetes deployment resources for Aetherveil Sentinel, including Helm charts and deployment scripts for production-ready deployment.

## Architecture

The Kubernetes deployment includes:

- **Coordinator**: Distributed swarm coordinator with leader election and consensus
- **Agents**: Specialized security agents (reconnaissance, scanner, exploiter, osint, stealth)
- **Dependencies**: Redis, PostgreSQL, monitoring stack
- **Security**: TLS certificates, RBAC, network policies, pod security policies
- **Monitoring**: Prometheus, Grafana, alerting rules
- **Tracing**: Jaeger for distributed tracing
- **Auto-scaling**: Horizontal Pod Autoscaler for dynamic scaling

## Prerequisites

- Kubernetes cluster (v1.25+)
- Helm 3.x
- kubectl configured
- cert-manager (for TLS certificates)
- Storage classes for persistent volumes

## Quick Start

### 1. Clone and Navigate

```bash
git clone <repository>
cd aetherveil_sentinel/deployment/kubernetes
```

### 2. Install for Development

```bash
# Install with development settings
./scripts/deploy.sh install -e development

# Check status
./scripts/deploy.sh status

# Access coordinator
./scripts/deploy.sh port-forward
```

### 3. Install for Production

```bash
# Update production values
cp environments/production.yaml my-production-values.yaml
# Edit my-production-values.yaml with your settings

# Install with production settings
./scripts/deploy.sh install -e production -f my-production-values.yaml

# Check status
./scripts/deploy.sh status
```

## Deployment Script

The `deploy.sh` script provides comprehensive deployment management:

```bash
# Installation
./scripts/deploy.sh install [options]

# Upgrade
./scripts/deploy.sh upgrade [options]

# Uninstall
./scripts/deploy.sh uninstall

# Status and monitoring
./scripts/deploy.sh status
./scripts/deploy.sh logs [component]
./scripts/deploy.sh test

# Development
./scripts/deploy.sh shell
./scripts/deploy.sh port-forward
```

### Script Options

- `-n, --namespace`: Kubernetes namespace (default: aetherveil-sentinel)
- `-e, --environment`: Environment (dev/staging/production)
- `-f, --values-file`: Custom values file
- `-t, --timeout`: Operation timeout (default: 600s)

## Configuration

### Environment Files

- `environments/development.yaml`: Development environment settings
- `environments/production.yaml`: Production environment settings

### Key Configuration Areas

#### Coordinator Configuration

```yaml
coordinator:
  replicaCount: 3
  autoscaling:
    enabled: true
    minReplicas: 2
    maxReplicas: 10
  persistence:
    enabled: true
    size: 50Gi
  ingress:
    enabled: true
    hosts:
      - host: aetherveil.company.com
```

#### Agent Configuration

```yaml
agents:
  reconnaissance:
    replicaCount: 3
    autoscaling:
      enabled: true
      minReplicas: 2
      maxReplicas: 8
    resources:
      limits:
        cpu: 1000m
        memory: 2Gi
```

#### Security Configuration

```yaml
security:
  enabled: true
  networkPolicy:
    enabled: true
  tls:
    enabled: true
  certificates:
    enabled: true
    issuer: letsencrypt-prod
```

#### Monitoring Configuration

```yaml
monitoring:
  enabled: true
  prometheus:
    enabled: true
    server:
      retention: "30d"
      persistentVolume:
        size: 100Gi
  grafana:
    enabled: true
    persistence:
      size: 20Gi
```

## Security Features

### Network Security

- **Network Policies**: Restrict network traffic between components
- **TLS/mTLS**: Encrypted communication between all components
- **Certificate Management**: Automated certificate lifecycle with cert-manager

### Access Control

- **RBAC**: Role-based access control for Kubernetes resources
- **Service Accounts**: Dedicated service accounts with minimal permissions
- **Pod Security Policies**: Enforced security policies for pod execution

### Secrets Management

- **Kubernetes Secrets**: Encrypted storage of sensitive data
- **Certificate Rotation**: Automated certificate rotation
- **Password Management**: Secure password generation and storage

## Monitoring and Observability

### Metrics

- **Prometheus**: Metrics collection and storage
- **Grafana**: Visualization and dashboards
- **Custom Metrics**: Application-specific metrics

### Alerting

- **PrometheusRule**: Comprehensive alerting rules
- **AlertManager**: Alert routing and notifications
- **Integration**: Slack, email, webhook notifications

### Logging

- **Centralized Logging**: ELK stack integration
- **Log Aggregation**: Fluent Bit for log shipping
- **Structured Logging**: JSON format for better parsing

### Tracing

- **Jaeger**: Distributed tracing
- **OpenTelemetry**: Standardized tracing instrumentation
- **Performance Monitoring**: Request flow analysis

## Scaling and High Availability

### Horizontal Pod Autoscaler

- **CPU/Memory Based**: Scale based on resource utilization
- **Custom Metrics**: Scale based on application metrics
- **Predictive Scaling**: Planned scaling events

### Pod Disruption Budgets

- **Availability Guarantees**: Ensure minimum replica count
- **Maintenance Windows**: Controlled updates and restarts
- **Chaos Engineering**: Resilience testing

### Leader Election

- **Distributed Consensus**: Raft-like consensus protocol
- **Automatic Failover**: Leader election on failure
- **Split-Brain Prevention**: Consistent cluster state

## Backup and Recovery

### Automated Backups

- **Scheduled Backups**: Daily backups to S3
- **Retention Policies**: Configurable retention periods
- **Incremental Backups**: Efficient storage usage

### Disaster Recovery

- **Multi-Region Support**: Cross-region replication
- **Point-in-Time Recovery**: Restore to specific timestamps
- **Backup Validation**: Automated backup testing

## Maintenance

### Updates and Upgrades

```bash
# Update Helm dependencies
helm dependency update ./helm/aetherveil-sentinel

# Upgrade deployment
./scripts/deploy.sh upgrade

# Rollback if needed
helm rollback aetherveil-sentinel
```

### Certificate Management

```bash
# Check certificate status
kubectl get certificates -n aetherveil-sentinel

# Force certificate renewal
kubectl delete certificate aetherveil-coordinator -n aetherveil-sentinel
```

### Scaling Operations

```bash
# Scale coordinator
kubectl scale deployment aetherveil-sentinel-coordinator --replicas=5

# Scale specific agent type
kubectl scale deployment aetherveil-sentinel-agent-reconnaissance --replicas=3
```

## Troubleshooting

### Common Issues

1. **Pods not starting**: Check resource limits and node capacity
2. **Certificate issues**: Verify cert-manager installation
3. **Network connectivity**: Check network policies and DNS
4. **Storage issues**: Verify storage class and PVC status

### Debug Commands

```bash
# Check pod status
kubectl get pods -n aetherveil-sentinel -o wide

# View pod logs
kubectl logs -n aetherveil-sentinel -l app.kubernetes.io/component=coordinator

# Describe resources
kubectl describe deployment aetherveil-sentinel-coordinator -n aetherveil-sentinel

# Check events
kubectl get events -n aetherveil-sentinel --sort-by=.metadata.creationTimestamp

# Access pod shell
kubectl exec -it <pod-name> -n aetherveil-sentinel -- /bin/bash
```

### Performance Monitoring

```bash
# Monitor resource usage
kubectl top pods -n aetherveil-sentinel
kubectl top nodes

# Check HPA status
kubectl get hpa -n aetherveil-sentinel

# Monitor metrics
kubectl port-forward svc/aetherveil-sentinel-coordinator 9090:9090
```

## Development Workflow

### Local Development

1. **Minikube Setup**: Use development values file
2. **Hot Reloading**: Configure volume mounts for development
3. **Debug Mode**: Enable debug logging and profiling

### CI/CD Integration

1. **Build Pipeline**: Automated image building
2. **Testing**: Integration and E2E tests
3. **Deployment**: Automated deployment to staging/production

### Testing

```bash
# Run deployment tests
./scripts/deploy.sh test

# Run specific component tests
kubectl run test-coordinator --image=curlimages/curl --rm -i --tty --restart=Never \
  -- curl -f http://aetherveil-sentinel-coordinator:8080/health
```

## Best Practices

### Resource Management

- Set appropriate resource requests and limits
- Use node selectors for workload placement
- Implement proper autoscaling policies

### Security

- Regular security audits
- Keep dependencies updated
- Use least privilege principles
- Enable security monitoring

### Operations

- Monitor resource usage and performance
- Implement proper logging and alerting
- Regular backup testing
- Disaster recovery procedures

## Support and Documentation

- **Architecture Documentation**: See `docs/architecture.md`
- **API Documentation**: See `docs/api.md`
- **Troubleshooting Guide**: See `docs/troubleshooting.md`
- **Security Guide**: See `docs/security.md`

## Contributing

1. Fork the repository
2. Create feature branch
3. Make changes and test thoroughly
4. Submit pull request with description

## License

This project is licensed under the MIT License. See LICENSE file for details.