# Aetherveil Autonomous DevOps Platform Configuration

# GCP Project Settings
project_id = "tidy-computing-465909-i3"
region     = "us-central1"
zone       = "us-central1-a"

# Deployment Phase Control
deploy_phase_1 = true
deploy_phase_2 = false
deploy_phase_3 = false

# Environment Configuration
environment = "production"

# Multi-Region Spoofing Configuration (Cost-Optimized)
enable_spoofing = true
spoofed_regions = ["europe-west1"]  # Reduced to single spoofed region for cost savings
domain_name     = "aetherveil.example.com"

# Service Account
service_account_email = "aetherveil-cicd@tidy-computing-465909-i3.iam.gserviceaccount.com"

# Secrets (use with care - consider using Secret Manager)
hackerone_scope_content = "{\"programs\":[]}"
slack_webhook_url       = "https://hooks.slack.com/services/placeholder"