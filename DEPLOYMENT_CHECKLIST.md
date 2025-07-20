# Aetherveil 3.0 Phase 1 - Deployment Checklist

## 🚀 Pre-Deployment Requirements

### ✅ Prerequisites
- [ ] GCP Project with billing enabled
- [ ] Domain name you control (update `terraform.tfvars`)
- [ ] gcloud CLI installed and authenticated (`gcloud auth login`)
- [ ] Project ID configured (`gcloud config set project YOUR_PROJECT_ID`)
- [ ] Terraform ≥1.7.0 installed (auto-installed by deploy script)

### ✅ Cost Controls
- [ ] Billing alerts configured (recommended: $300 budget)
- [ ] Resource quotas reviewed
- [ ] Expected monthly cost: $180-365 (verified in COST_OPTIMIZATION_SUMMARY.md)

---

## 🛠️ Deployment Steps

### 1. Configuration
```bash
# Clone and navigate to repository
cd /root/Aetherveil

# Update domain in terraform.tfvars
vim infra/terraform/terraform.tfvars
# Set: domain_name = "yourdomain.com"

# Verify configuration
cat infra/terraform/terraform.tfvars
```

### 2. Deploy Infrastructure
```bash
# Run cost-optimized deployment
./deploy-phase1.sh

# Expected output:
# ✅ Real infrastructure deployed to us-central1
# ✅ Spoofing infrastructure created (europe-west1)
# ✅ Global Anycast IP assigned
# ✅ Regional DNS configured
# ✅ SSL certificates requested
# ✅ Synthetic audit logging enabled
```

### 3. Domain Configuration
```bash
# Get Cloud DNS nameservers
cd infra/terraform
terraform output spoofing_info | jq -r '.dns_zone'

# Configure your domain registrar:
# Point nameservers to Cloud DNS nameservers
# Example: ns-cloud-a1.googledomains.com
```

### 4. Wait for Propagation
- **DNS Propagation**: 24-48 hours globally
- **SSL Certificates**: 24-48 hours for managed certificates
- **CDN Cache**: 1-2 hours for initial warming

### 5. Verification
```bash
# After 24-48 hours, run verification
./spoofing_verification.sh

# Expected results:
# ✅ DNS resolution for regional endpoints
# ✅ Regional HTTP headers (X-Served-From, etc.)
# ✅ SSL certificate covering all domains
# ✅ CDN edge presence
# ✅ Synthetic audit logs in BigQuery
```

---

## 🔍 Post-Deployment Validation

### Infrastructure Verification
```bash
# Check Cloud Run services
gcloud run services list --platform=managed

# Verify global IP assignment
gcloud compute addresses list --global

# Check DNS zone creation
gcloud dns managed-zones list

# Verify SSL certificate status
gcloud compute ssl-certificates list --global
```

### Cost Verification
```bash
# Check current resource usage
gcloud compute instances list  # Should show minimal/no VMs
gcloud storage buckets list    # Should show cost-optimized buckets
gcloud run services list       # Should show scale-to-zero configuration

# Monitor billing
gcloud billing projects describe $PROJECT_ID
```

### Spoofing Effectiveness Test
```bash
# Test regional DNS resolution
dig us-central1.yourdomain.com
dig europe-west1.yourdomain.com

# Test regional headers
curl -H "Host: europe-west1.yourdomain.com" https://GLOBAL_IP/ -I

# Verify synthetic logs
bq query --use_legacy_sql=false \
  "SELECT region, COUNT(*) FROM \`$PROJECT_ID.security_analytics.synthetic_audit_logs\` 
   WHERE timestamp > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 DAY) 
   GROUP BY region"
```

---

## 📊 Success Metrics

### Cost Targets
- [ ] Monthly cost < $365 (target: $180-365)
- [ ] No unexpected charges from terminated resources
- [ ] Storage costs < $20/month with lifecycle policies
- [ ] Compute costs < $100/month with scale-to-zero

### Spoofing Effectiveness
- [ ] All regional subdomains resolve to same global IP
- [ ] Regional HTTP headers present (X-Served-From, etc.)
- [ ] SSL certificate covers all domains
- [ ] Synthetic audit logs generating every 6 hours
- [ ] Dummy infrastructure visible in regional dashboards

### Security Compliance
- [ ] CMEK encryption for sensitive data
- [ ] IAM least-privilege service accounts
- [ ] DNSSEC enabled for domain
- [ ] VPC security (serverless VPC access)

---

## 🛠️ Troubleshooting

### Common Issues

#### SSL Certificate Not Provisioning
```bash
# Check certificate status
gcloud compute ssl-certificates describe aetherveil-cert --global

# Common fix: Verify domain ownership
# Ensure nameservers point to Cloud DNS
```

#### DNS Not Resolving
```bash
# Check DNS configuration
gcloud dns managed-zones describe aetherveil-zone

# Verify nameserver delegation
dig NS yourdomain.com

# Wait 24-48 hours for global propagation
```

#### High Costs
```bash
# Check resource usage
gcloud compute instances list --filter="status=RUNNING"
gcloud run services list --format="table(metadata.name,status.traffic[0].percent)"

# Verify scale-to-zero configuration
gcloud run services describe SERVICE_NAME --region=us-central1 \
  --format="value(spec.template.metadata.annotations['autoscaling.knative.dev/minScale'])"
```

#### Regional Headers Missing
```bash
# Check Cloud Run service status
gcloud run services describe pipeline-observer --region=us-central1

# Verify load balancer configuration
gcloud compute url-maps describe aetherveil-lb --global

# Check backend health
gcloud compute backend-services get-health aetherveil-backend --global
```

#### Synthetic Logs Not Generating
```bash
# Check Cloud Scheduler job
gcloud scheduler jobs describe synthetic-audit-generator --location=us-central1

# Verify service account permissions
gcloud projects get-iam-policy $PROJECT_ID \
  --flatten="bindings[].members" \
  --filter="bindings.members:*pipeline-observer*"
```

### Emergency Cost Controls
```bash
# Immediately scale down if costs spike
gcloud run services update pipeline-observer \
  --region=us-central1 \
  --min-instances=0 \
  --max-instances=1

# Disable synthetic log generation
gcloud scheduler jobs pause synthetic-audit-generator --location=us-central1

# Check and delete expensive resources
gcloud compute instances list --filter="status=RUNNING"
gcloud container clusters list
```

---

## 📈 Monitoring & Maintenance

### Daily Checks
- [ ] Billing dashboard review
- [ ] Cloud Run service health
- [ ] Synthetic log generation
- [ ] SSL certificate validity

### Weekly Reviews
- [ ] Cost optimization opportunities
- [ ] Security scan results
- [ ] Performance metrics
- [ ] Regional spoofing effectiveness

### Monthly Tasks
- [ ] BigQuery data cleanup (30-day retention)
- [ ] Storage lifecycle policy effectiveness
- [ ] Domain certificate renewal status
- [ ] Budget and quota adjustments

---

## 🚀 Next Phase Planning

### Phase 2 Migration Triggers
- [ ] Monthly budget increases to >$1,000
- [ ] Real multi-region traffic requirements
- [ ] Compliance requirements for data residency
- [ ] Performance needs exceed spoofing capabilities

### Phase 2 Preparation
```bash
# Enable Phase 2 components
cd infra/terraform
terraform plan -var="deploy_phase_2=true"

# Migration to real multi-region
./deploy-phase2.sh --migrate-from-spoofing
```

---

## 📞 Support Resources

### Documentation
- `README_PHASE1_SPOOFING.md` - Architecture overview
- `COST_OPTIMIZATION_SUMMARY.md` - Detailed cost breakdown
- `spoofing_verification.sh` - Verification script

### Monitoring
- **GCP Console**: https://console.cloud.google.com
- **Cloud Run**: https://console.cloud.google.com/run
- **BigQuery**: https://console.cloud.google.com/bigquery
- **Billing**: https://console.cloud.google.com/billing

### Emergency Contacts
- Review GCP support plan for critical issues
- Monitor GitHub Issues for known problems
- Check Cloud Status: https://status.cloud.google.com

---

## ✅ Deployment Complete Checklist

- [ ] Infrastructure deployed successfully
- [ ] Domain nameservers configured  
- [ ] SSL certificates provisioning
- [ ] Regional spoofing functional
- [ ] Cost monitoring active
- [ ] Verification script passes
- [ ] Documentation reviewed
- [ ] Monitoring alerts configured

**🎯 Expected Result**: Cost-effective (<$365/month) single-region deployment that convincingly simulates multi-region presence for red-team operations.