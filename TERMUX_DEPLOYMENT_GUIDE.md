# 🚀 Aetherveil Autonomous AI DevOps Platform - Termux Deployment Guide

## 📱 Termux Ubuntu (proot-distro) Cloud-Native Deployment

This guide provides step-by-step instructions for deploying the Aetherveil Autonomous AI-Powered DevOps Platform in a Termux Ubuntu environment without Docker, using GCP Cloud Build for all container builds.

---

## 🔧 Prerequisites

### ✅ Environment Setup
```bash
# Ensure you're in Termux Ubuntu proot environment
proot-distro login ubuntu

# Verify gcloud CLI is installed and authenticated
gcloud auth list
gcloud config get-value project

# Set project if not already set
export PROJECT_ID="tidy-computing-465909-i3"
gcloud config set project $PROJECT_ID
```

### ✅ Required APIs and Services
All required APIs should already be enabled:
- ✅ Cloud Build API (`cloudbuild.googleapis.com`)
- ✅ Artifact Registry API (`artifactregistry.googleapis.com`)
- ✅ Cloud Run API (`run.googleapis.com`)
- ✅ Firestore API (`firestore.googleapis.com`)
- ✅ BigQuery API (`bigquery.googleapis.com`)
- ✅ Pub/Sub API (`pubsub.googleapis.com`)

---

## 🚀 Quick Deployment (Recommended)

### Option 1: One-Command Full Deployment
```bash
# Deploy all phases at once
./deploy.sh --phase 3

# This will:
# 1. Build all agent containers via Cloud Build
# 2. Deploy Terraform infrastructure 
# 3. Set up Firestore and BigQuery
# 4. Deploy Cloud Functions
# 5. Configure monitoring and alerting
```

### Option 2: Phase-by-Phase Deployment
```bash
# Deploy Phase 1 only (immediate deployment)
./deploy.sh --phase 1

# Later, deploy Phase 2 (medium-term)
./deploy.sh --phase 2

# Finally, deploy Phase 3 (long-term)
./deploy.sh --phase 3
```

---

## 🔨 Manual Build Process (Alternative)

If you prefer to build agents separately before deployment:

### Step 1: Build All Agents
```bash
# Build all phases at once
./build_agents.sh all

# OR build specific phases
./build_agents.sh phase 1
./build_agents.sh phase 2
```

### Step 2: Monitor Build Progress
```bash
# Check build status
./build_agents.sh status

# Wait for builds to complete
./build_agents.sh wait

# View built images
./build_agents.sh images
```

### Step 3: Deploy Infrastructure
```bash
# Deploy with pre-built images
./deploy.sh --phase <1|2|3>
```

---

## 📋 Detailed Step-by-Step Deployment

### Phase 1: Immediate Deployment (0-3 months)

```bash
# 1. Build Phase 1 agents
log "Building Phase 1 agents..."
./build_agents.sh phase 1

# 2. Deploy infrastructure
log "Deploying Phase 1 infrastructure..."
cd infra/terraform
terraform init -backend-config="bucket=aetherveil-terraform-state-${PROJECT_ID}"
terraform plan -var="project_id=${PROJECT_ID}" -var="region=us-central1" -var="deploy_phase_1=true"
terraform apply -auto-approve

# 3. Deploy Cloud Functions
log "Deploying webhook processor..."
gcloud functions deploy github-webhook-processor \
    --gen2 \
    --runtime=python311 \
    --region=us-central1 \
    --source=functions/webhook-processor \
    --entry-point=process_webhook \
    --trigger-http \
    --allow-unauthenticated

# 4. Verify deployment
gcloud run services list
gcloud artifacts docker images list us-central1-docker.pkg.dev/${PROJECT_ID}/aetherveil
```

### Phase 2: Medium-term Deployment (3-9 months)

```bash
# 1. Build Phase 2 agents (includes Phase 1)
./build_agents.sh phase 2

# 2. Update infrastructure for Phase 2
cd infra/terraform
terraform plan -var="deploy_phase_2=true"
terraform apply -auto-approve

# 3. Configure OpenAI API key
echo "your-openai-api-key" | gcloud secrets create openai-api-key --data-file=-

# 4. Verify ADO deployment
gcloud run services describe ado-orchestrator --region=us-central1
```

### Phase 3: Long-term Deployment (9-18 months)

```bash
# Phase 3 agents will be implemented in future releases
# Current architecture is ready for Phase 3 expansion
log "Phase 3 infrastructure ready for future implementation"
```

---

## 🔍 Troubleshooting

### Common Issues and Solutions

#### Build Failures
```bash
# Check Cloud Build logs
gcloud builds list --limit=10
gcloud builds log <BUILD_ID>

# Retry failed builds
./build_agents.sh cleanup
./build_agents.sh phase <phase_number>
```

#### Authentication Issues
```bash
# Re-authenticate if needed
gcloud auth login
gcloud auth application-default login

# Verify permissions
gcloud projects get-iam-policy ${PROJECT_ID}
```

#### Terraform Issues
```bash
# Reset Terraform state if needed
cd infra/terraform
terraform state list
terraform refresh

# Force unlock if state is locked
terraform force-unlock <LOCK_ID>
```

#### Missing APIs
```bash
# Enable any missing APIs
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable firestore.googleapis.com
```

---

## 🎯 Verification Commands

### Check Deployment Status
```bash
# Cloud Run services
gcloud run services list --region=us-central1

# Artifact Registry images
gcloud artifacts docker images list us-central1-docker.pkg.dev/${PROJECT_ID}/aetherveil

# Firestore database
gcloud firestore databases describe --database="(default)"

# BigQuery datasets
bq ls

# Pub/Sub topics
gcloud pubsub topics list

# Cloud Functions
gcloud functions list --region=us-central1
```

### Test Agent Health
```bash
# Pipeline Observer health check
curl -X GET "https://pipeline-observer-<hash>-uc.a.run.app/health"

# ADO Orchestrator health check (Phase 2)
curl -X GET "https://ado-orchestrator-<hash>-uc.a.run.app/health"

# Check agent status
curl -X GET "https://ado-orchestrator-<hash>-uc.a.run.app/agents/status"
```

---

## 📊 Monitoring and Observability

### Cloud Console Links
- **Cloud Build**: https://console.cloud.google.com/cloud-build/builds
- **Cloud Run**: https://console.cloud.google.com/run
- **Artifact Registry**: https://console.cloud.google.com/artifacts
- **Firestore**: https://console.cloud.google.com/firestore
- **Monitoring**: https://console.cloud.google.com/monitoring

### Custom Dashboards
```bash
# Import monitoring dashboard
gcloud monitoring dashboards create --config-from-file=monitoring/autonomous-platform-dashboard.json
```

---

## 💰 Cost Optimization

### Estimated Monthly Costs (Termux Deployment)
- **Phase 1**: $95-210/month
  - Cloud Run: $50-100
  - BigQuery: $20-50  
  - Firestore: $10-25
  - Pub/Sub: $10-20
  - Cloud Functions: $5-15

- **Phase 2 Additional**: $250-650/month
  - LLM API calls: $100-300
  - Vertex AI: $50-150
  - GKE: $100-200

### Cost Monitoring
```bash
# Set up billing alerts
gcloud alpha billing budgets create \
    --billing-account=<BILLING_ACCOUNT_ID> \
    --display-name="Aetherveil Platform Budget" \
    --budget-amount=500USD

# Monitor usage
gcloud logging read "resource.type=cloud_run_revision" --limit=100
```

---

## 🔄 CI/CD Integration

### GitHub Actions Integration
```bash
# Configure Workload Identity Federation (already set up)
# Workload Identity Provider: projects/959633659546/locations/global/workloadIdentityPools/github-actions-pool/providers/github-provider
# Service Account: aetherveil-cicd@tidy-computing-465909-i3.iam.gserviceaccount.com

# Webhook URL for GitHub
echo "Configure GitHub webhook to:"
gcloud functions describe github-webhook-processor --region=us-central1 --format="value(httpsTrigger.url)"
```

---

## 🎉 Success Metrics

After successful deployment, you should see:

### ✅ Infrastructure
- 3-4 Cloud Run services running (depending on phase)
- Firestore database with collections created
- BigQuery datasets with tables
- Pub/Sub topics configured
- Cloud Functions deployed

### ✅ Monitoring
- Custom metrics flowing to Cloud Monitoring
- Alert policies configured
- Dashboards displaying real-time data

### ✅ Autonomous Capabilities
- Pipeline anomaly detection active
- Self-healing workflows operational
- Performance baselines being established
- (Phase 2) Multi-agent LLM decision making

---

## 🆘 Support

### Getting Help
```bash
# View deployment logs
./deploy.sh --help

# Check build script options
./build_agents.sh --help

# View recent operations
gcloud logging read "timestamp>=2024-01-01" --limit=50
```

### Useful Commands Reference
```bash
# Quick status check
gcloud run services list && gcloud builds list --limit=5

# Restart services
gcloud run services update SERVICE_NAME --region=us-central1

# View logs
gcloud logs read "resource.type=cloud_run_revision" --limit=50

# Scale services
gcloud run services update SERVICE_NAME --max-instances=10 --region=us-central1
```

---

## 🔮 Next Steps

1. **Configure GitHub Webhooks**: Point your repository webhooks to the deployed Cloud Function
2. **Set up Monitoring**: Configure alert notification channels
3. **API Keys**: Add OpenAI/Anthropic API keys to Secret Manager for Phase 2
4. **Testing**: Run your first pipeline to see autonomous agents in action
5. **Scaling**: Monitor usage and adjust Cloud Run configurations as needed

---

*The Aetherveil Autonomous AI-Powered DevOps Platform is now fully deployed and operational in your Termux environment! 🎯*