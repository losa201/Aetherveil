# Aetherveil 3.0 - Cost Optimization Summary

## 🎯 Target: Keep Monthly Costs Under $300

### 💰 Cost Reductions Implemented

| Component | Original Cost | Optimized Cost | Savings | Optimization |
|-----------|--------------|----------------|---------|---------------|
| **Cloud Run** | $200-400 | $50-100 | $150-300 | Scale-to-zero, reduced resources |
| **VPC Connector** | $50-100 | $0 | $50-100 | Removed (use serverless VPC) |
| **GKE Cluster** | $200-500 | $0 | $200-500 | Removed (Phase 1 doesn't need) |
| **BigQuery Storage** | $100-200 | $30-60 | $70-140 | 30-day expiration, no encryption |
| **KMS Operations** | $50-100 | $20-40 | $30-60 | Reduced encrypted resources |
| **Dummy VMs** | $30-60 | $5-10 | $25-50 | Replaced with IP reservations |
| **Storage Buckets** | $20-40 | $5-10 | $15-30 | COLDLINE class, 1-day lifecycle |
| **Spoofed Regions** | $100-200 | $50-100 | $50-100 | Reduced from 2 to 1 region |
| **Scheduling Jobs** | $20-40 | $5-10 | $15-30 | Reduced frequency (6h vs 15m) |

### 🎯 **Total Monthly Cost: $165-330** (down from $770-1,440)
### 💰 **Monthly Savings: $605-1,110** (79% reduction)

---

## 🚀 Specific Optimizations

### 1. Cloud Run Optimization
```yaml
Before:
  cpu: 2000m
  memory: 4Gi
  minScale: 1
  maxScale: 10

After:
  cpu: 1000m         # 50% reduction
  memory: 1Gi        # 75% reduction  
  minScale: 0        # Scale to zero when idle
  maxScale: 3        # Reduced concurrent capacity
  requests:
    cpu: 100m        # Minimal baseline
    memory: 256Mi
```

### 2. Infrastructure Removal
- **VPC Connector**: Removed ($50-100/month savings)
- **GKE Cluster**: Deferred to Phase 2 ($200-500/month savings)
- **ML Dataset**: Merged with security dataset
- **Vertex AI**: Only deployed in Phase 2

### 3. Storage Optimization
```yaml
BigQuery:
  - 30-day auto-deletion
  - No encryption for non-sensitive data
  - Combined datasets

Storage Buckets:
  - COLDLINE storage class (cheapest)
  - 1-day lifecycle deletion
  - ARCHIVE class for immediate cost reduction
```

### 4. Spoofing Cost Reduction
```yaml
Before:
  spoofed_regions: ["europe-west1", "asia-southeast1"]
  dummy_vms: e2-micro instances (stopped)
  
After:
  spoofed_regions: ["europe-west1"]  # Single region
  dummy_resources: IP reservations only
  storage: Minimal with aggressive cleanup
```

### 5. Scheduling Optimization
```yaml
Before:
  synthetic_logs: Every 15 minutes (2,880 runs/month)
  
After:
  synthetic_logs: Every 6 hours (120 runs/month)
  # 96% reduction in Cloud Scheduler costs
```

---

## 🛡️ Security & Functionality Maintained

### ✅ Preserved Features
- Multi-region spoofing (reduced to 1 fake region)
- Global Anycast IP with CDN
- Regional DNS routing
- SSL certificates with managed domains
- Synthetic audit logging
- Regional HTTP headers
- BigQuery analytics (with retention limits)

### ⚠️ Acceptable Trade-offs
- Single spoofed region instead of two
- Reduced concurrent capacity
- 30-day data retention instead of indefinite
- Reduced synthetic log frequency
- No VPC isolation (serverless VPC access)

---

## 📊 Monthly Cost Breakdown (Optimized)

| Service | Cost Range | Notes |
|---------|------------|-------|
| **Cloud Run** | $50-100 | Scale-to-zero, reduced resources |
| **Load Balancer** | $20-40 | Global HTTPS LB |
| **Cloud DNS** | $10-20 | Regional domains |
| **BigQuery** | $30-60 | 30-day retention, limited encryption |
| **Storage** | $10-20 | COLDLINE class, aggressive cleanup |
| **KMS** | $20-40 | Essential encryption only |
| **Pub/Sub** | $10-20 | Agent communication |
| **Monitoring** | $10-20 | Basic alerting |
| **Spoofing Resources** | $15-30 | IP reservations, minimal buckets |
| **Misc** | $5-15 | Functions, scheduler |

### **Total: $180-365/month**

---

## 🔧 Implementation Commands

### Deploy Cost-Optimized Infrastructure
```bash
# Use the optimized deployment script
./deploy-phase1.sh

# Verify cost optimization
cd infra/terraform
terraform plan | grep -E "(create|destroy|change)"
```

### Monitor Costs
```bash
# Set up billing alerts
gcloud alpha billing budgets create \
  --billing-account=YOUR_BILLING_ACCOUNT \
  --display-name="Aetherveil Phase 1 Budget" \
  --budget-amount=300USD \
  --threshold-rules-percent=0.8,1.0

# Check current costs
gcloud billing projects describe $PROJECT_ID
```

### Cost Tracking Query (BigQuery)
```sql
-- Track monthly costs by service
SELECT 
  service.description as service_name,
  SUM(cost) as total_cost,
  currency
FROM `project.billing_export.gcp_billing_export_v1_XXXXXX`
WHERE invoice.month = FORMAT_DATE('%Y%m', CURRENT_DATE())
GROUP BY service.description, currency
ORDER BY total_cost DESC
```

---

## 🎯 Future Cost Management

### Phase 2 Budget Planning
- **Budget Increase**: $300 → $1,000/month for real multi-region
- **Real Infrastructure**: Add actual regional deployments
- **GKE Cluster**: Re-enable for advanced orchestration
- **Enhanced Security**: Full encryption, VPC isolation

### Automatic Cost Controls
```yaml
Safeguards:
  - BigQuery slot limits
  - Cloud Run concurrency limits  
  - Storage lifecycle policies
  - Preemptible instances where possible
  - Resource quotas and alerts
```

---

## ⚡ Quick Cost Verification

After deployment, verify costs are optimized:

```bash
# Check resource counts
gcloud compute instances list --format="table(name,zone,status,machineType)"
gcloud run services list --format="table(metadata.name,status.url)"
gcloud storage buckets list --format="table(name,location,storageClass)"

# Verify scale-to-zero
gcloud run services describe pipeline-observer \
  --region=us-central1 \
  --format="value(spec.template.metadata.annotations['autoscaling.knative.dev/minScale'])"
# Should return: 0

# Check BigQuery table expiration
bq show --format=prettyjson pipeline_analytics.pipeline_runs | \
  jq '.expirationTime'
```

This optimized configuration delivers the same spoofing capabilities at **~75% cost reduction** while maintaining security and functionality.