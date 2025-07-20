# Autonomous AI-Powered DevOps Platform Infrastructure
# Complete multi-phase deployment for Aetherveil

terraform {
  required_version = ">= 1.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = "~> 5.0"
    }
  }
  
  backend "gcs" {
    bucket = "aetherveil-terraform-state-tidy-computing-465909-i3"
    prefix = "autonomous-devops"
  }
}

# Variable declarations
variable "project_id" {
  description = "GCP Project ID"
  type        = string
  default     = "tidy-computing-465909-i3"
}

variable "region" {
  description = "GCP Region"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "GCP Zone"
  type        = string
  default     = "us-central1-a"
}

variable "spoofed_regions" {
  description = "Regions to simulate presence in (for spoofing)"
  type        = list(string)
  default     = ["europe-west1", "asia-southeast1"]
}

variable "domain_name" {
  description = "Base domain for multi-region spoofing"
  type        = string
  default     = "aetherveil.com"
}

variable "enable_spoofing" {
  description = "Enable multi-region spoofing mechanisms"
  type        = bool
  default     = true
}

variable "deploy_phase_1" {
  description = "Deploy Phase 1 components"
  type        = bool
  default     = true
}

variable "deploy_phase_2" {
  description = "Deploy Phase 2 components"
  type        = bool
  default     = false
}

variable "deploy_phase_3" {
  description = "Deploy Phase 3 components"
  type        = bool
  default     = false
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}

variable "service_account_email" {
  description = "Email of the service account used for CI/CD deployments."
  type        = string
}

variable "hackerone_scope_content" {
  description = "Content of the HackerOne scope file"
  type        = string
  default     = "{}"
}

variable "slack_webhook_url" {
  description = "Slack webhook URL for notifications"
  type        = string
  default     = "https://hooks.slack.com/services/placeholder"
}

locals {
  project_id = var.project_id
  region     = var.region
  zone       = var.zone
  
  # Phase deployment flags
  deploy_phase_1 = var.deploy_phase_1
  deploy_phase_2 = var.deploy_phase_2
  deploy_phase_3 = var.deploy_phase_3
  
  # Spoofing configuration
  spoofed_regions = var.spoofed_regions
  all_regions     = concat([var.region], var.spoofed_regions)
  
  # Regional mappings for spoofing
  region_mapping = {
    "us-central1"     = { timezone = "America/Chicago", latency_base = 0 }
    "europe-west1"    = { timezone = "Europe/London", latency_base = 120 }
    "asia-southeast1" = { timezone = "Asia/Singapore", latency_base = 200 }
  }
}

# Project configuration
provider "google" {
  project = local.project_id
  region  = local.region
}

provider "google-beta" {
  project = local.project_id
  region  = local.region
}

# Enable required APIs
resource "google_project_service" "apis" {
  for_each = toset([
    "run.googleapis.com",
    "cloudbuild.googleapis.com",
    "artifactregistry.googleapis.com",
    "pubsub.googleapis.com",
    "bigquery.googleapis.com",
    "firestore.googleapis.com",
    "aiplatform.googleapis.com",
    "ml.googleapis.com",
    "cloudscheduler.googleapis.com",
    "cloudfunctions.googleapis.com",
    "monitoring.googleapis.com",
    "logging.googleapis.com",
    "secretmanager.googleapis.com",
    "container.googleapis.com",
    "compute.googleapis.com",
    "cloudkms.googleapis.com",
    "securitycenter.googleapis.com",
    "binaryauthorization.googleapis.com",
    "dns.googleapis.com",
    "certificatemanager.googleapis.com"
  ])
  
  service = each.value
  project = local.project_id
}

# KMS for encryption
resource "google_kms_key_ring" "aetherveil" {
  name     = "aetherveil-keyring"
  location = "global"
  project  = local.project_id
}

resource "google_kms_crypto_key" "main" {
  name     = "aetherveil-main-key"
  key_ring = google_kms_key_ring.aetherveil.id
  purpose  = "ENCRYPT_DECRYPT"
  
  lifecycle {
    prevent_destroy = true
  }
}

# === MULTI-REGION SPOOFING INFRASTRUCTURE ===

# Global static IP for Anycast
resource "google_compute_global_address" "aetherveil_global" {
  count        = var.enable_spoofing ? 1 : 0
  name         = "aetherveil-global-ip"
  address_type = "EXTERNAL"
  ip_version   = "IPV4"
}

# Cloud DNS managed zone
resource "google_dns_managed_zone" "aetherveil_zone" {
  count       = var.enable_spoofing ? 1 : 0
  name        = "aetherveil-zone"
  dns_name    = "${var.domain_name}."
  description = "Aetherveil multi-region DNS zone"
  
  dnssec_config {
    state = "on"
  }
}

# Regional DNS records for spoofing
resource "google_dns_record_set" "regional_a_records" {
  count        = var.enable_spoofing ? length(local.all_regions) : 0
  managed_zone = google_dns_managed_zone.aetherveil_zone[0].name
  name         = "${local.all_regions[count.index]}.${var.domain_name}."
  type         = "A"
  ttl          = 300
  
  rrdatas = [google_compute_global_address.aetherveil_global[0].address]
}

# Cloud CDN for global edge presence
resource "google_compute_backend_service" "aetherveil_backend" {
  count                           = var.enable_spoofing ? 1 : 0
  name                           = "aetherveil-backend"
  protocol                       = "HTTP"
  timeout_sec                    = 30
  enable_cdn                     = true
  load_balancing_scheme          = "EXTERNAL_MANAGED"
  
  cdn_policy {
    cache_mode       = "CACHE_ALL_STATIC"
    default_ttl      = 3600
    max_ttl          = 86400
    negative_caching = true
    signed_url_cache_max_age_sec = 7200
    
    negative_caching_policy {
      code = 404
      ttl  = 120
    }
  }
  
  backend {
    group = google_compute_region_network_endpoint_group.aetherveil_neg[0].id
  }
}

# Network Endpoint Group for Cloud Run
resource "google_compute_region_network_endpoint_group" "aetherveil_neg" {
  count                 = var.enable_spoofing ? 1 : 0
  name                  = "aetherveil-neg"
  network_endpoint_type = "SERVERLESS"
  region                = local.region
  
  cloud_run {
    service = google_cloud_run_service.pipeline_observer[0].name
  }
}

# Global HTTPS Load Balancer
resource "google_compute_url_map" "aetherveil_lb" {
  count           = var.enable_spoofing ? 1 : 0
  name            = "aetherveil-lb"
  default_service = google_compute_backend_service.aetherveil_backend[0].id
  
  # Regional routing for spoofing
  dynamic "host_rule" {
    for_each = local.all_regions
    content {
      hosts        = ["${host_rule.value}.${var.domain_name}"]
      path_matcher = "regional-${host_rule.value}"
    }
  }
  
  dynamic "path_matcher" {
    for_each = local.all_regions
    content {
      name            = "regional-${path_matcher.value}"
      default_service = google_compute_backend_service.aetherveil_backend[0].id
      
      # Note: Regional headers will be added at application level for cost optimization
    }
  }
}

# HTTPS target proxy
resource "google_compute_target_https_proxy" "aetherveil_https_proxy" {
  count   = var.enable_spoofing ? 1 : 0
  name    = "aetherveil-https-proxy"
  url_map = google_compute_url_map.aetherveil_lb[0].id
  
  ssl_certificates = [google_compute_managed_ssl_certificate.aetherveil_cert[0].id]
}

# Managed SSL certificate
resource "google_compute_managed_ssl_certificate" "aetherveil_cert" {
  count = var.enable_spoofing ? 1 : 0
  name  = "aetherveil-cert"
  
  managed {
    domains = concat(
      ["${var.domain_name}"],
      [for region in local.all_regions : "${region}.${var.domain_name}"]
    )
  }
}

# Global forwarding rule
resource "google_compute_global_forwarding_rule" "aetherveil_https_forward" {
  count       = var.enable_spoofing ? 1 : 0
  name        = "aetherveil-https-forward"
  target      = google_compute_target_https_proxy.aetherveil_https_proxy[0].id
  port_range  = "443"
  ip_address  = google_compute_global_address.aetherveil_global[0].address
}

# HTTP to HTTPS redirect
resource "google_compute_url_map" "aetherveil_http_redirect" {
  count = var.enable_spoofing ? 1 : 0
  name  = "aetherveil-http-redirect"
  
  default_url_redirect {
    https_redirect         = true
    redirect_response_code = "MOVED_PERMANENTLY_DEFAULT"
    strip_query            = false
  }
}

resource "google_compute_target_http_proxy" "aetherveil_http_proxy" {
  count   = var.enable_spoofing ? 1 : 0
  name    = "aetherveil-http-proxy"
  url_map = google_compute_url_map.aetherveil_http_redirect[0].id
}

resource "google_compute_global_forwarding_rule" "aetherveil_http_forward" {
  count      = var.enable_spoofing ? 1 : 0
  name       = "aetherveil-http-forward"
  target     = google_compute_target_http_proxy.aetherveil_http_proxy[0].id
  port_range = "80"
  ip_address = google_compute_global_address.aetherveil_global[0].address
}

# === DUMMY INFRASTRUCTURE FOR SPOOFING ===

# Dummy KMS keyrings in spoofed regions
resource "google_kms_key_ring" "spoofed_region_keyrings" {
  for_each = var.enable_spoofing ? toset(local.spoofed_regions) : toset([])
  name     = "aetherveil-keyring-${each.value}"
  location = each.value
  project  = local.project_id
}

# Cost-optimized: Use only reserved IP addresses instead of VMs
resource "google_compute_address" "spoofed_region_ips" {
  for_each     = var.enable_spoofing ? toset(local.spoofed_regions) : toset([])
  name         = "aetherveil-dummy-ip-${each.value}"
  region       = each.value
  address_type = "INTERNAL"
  purpose      = "GCE_ENDPOINT"
  
  # Internal IP costs almost nothing but shows regional presence
}

# Cost-optimized: Minimal storage buckets with aggressive lifecycle
resource "google_storage_bucket" "spoofed_region_buckets" {
  for_each                    = var.enable_spoofing ? toset(local.spoofed_regions) : toset([])
  name                        = "${local.project_id}-dummy-${each.value}"
  location                    = upper(each.value)
  force_destroy              = true
  uniform_bucket_level_access = true
  storage_class              = "COLDLINE"  # Cheapest storage class
  
  # No encryption to save KMS costs - dummy bucket anyway
  
  lifecycle_rule {
    condition {
      age = 1  # Delete after 1 day to minimize storage costs
    }
    action {
      type = "Delete"
    }
  }
  
  lifecycle_rule {
    condition {
      age = 0
    }
    action {
      type          = "SetStorageClass"
      storage_class = "ARCHIVE"  # Immediate archival
    }
  }
}

# Secret Manager for HackerOne Scope
resource "google_secret_manager_secret" "hackerone_scope" {
  project     = local.project_id
  secret_id   = "hackerone-scope"
  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_version" "hackerone_scope_version" {
  secret      = google_secret_manager_secret.hackerone_scope.id
  secret_data = var.hackerone_scope_content
}

# Secret Manager for Slack Webhook URL
resource "google_secret_manager_secret" "slack_red_team_webhook" {
  project     = local.project_id
  secret_id   = "slack-red-team-webhook"
  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_version" "slack_red_team_webhook_version" {
  secret      = google_secret_manager_secret.slack_red_team_webhook.id
  secret_data = var.slack_webhook_url
}

# Service accounts for different components
resource "google_service_account" "pipeline_observer" {
  account_id   = "pipeline-observer"
  display_name = "Pipeline Observer Agent"
  project      = local.project_id
}

resource "google_service_account" "anomaly_detector" {
  account_id   = "anomaly-detector"
  display_name = "Log Anomaly Detector"
  project      = local.project_id
}

resource "google_service_account" "performance_analyzer" {
  account_id   = "performance-analyzer"
  display_name = "Performance Baseline Generator"
  project      = local.project_id
}

resource "google_service_account" "healing_engine" {
  account_id   = "healing-engine"
  display_name = "Self-Healing Workflow Engine"
  project      = local.project_id
}

resource "google_service_account" "security_gate" {
  account_id   = "security-gate"
  display_name = "Intelligent Security Gate"
  project      = local.project_id
}

resource "google_service_account" "red_team_pentester" {
  account_id   = "red-team-pentester"
  display_name = "Red Team Pentester Agent"
  project      = local.project_id
}

resource "google_service_account" "ado_orchestrator" {
  count        = local.deploy_phase_2 ? 1 : 0
  account_id   = "ado-orchestrator"
  display_name = "Autonomous DevOps Orchestrator"
  project      = local.project_id
}

# IAM bindings for service accounts
resource "google_project_iam_member" "pipeline_observer_permissions" {
  for_each = toset([
    "roles/bigquery.dataEditor",
    "roles/bigquery.jobUser",
    "roles/pubsub.publisher",
    "roles/pubsub.subscriber",
    "roles/logging.viewer",
    "roles/monitoring.metricWriter",
    "roles/datastore.user"
  ])
  
  project = local.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.pipeline_observer.email}"
}

resource "google_project_iam_member" "anomaly_detector_permissions" {
  for_each = toset([
    "roles/bigquery.dataEditor",
    "roles/bigquery.jobUser",
    "roles/pubsub.publisher",
    "roles/pubsub.subscriber",
    "roles/datastore.user",
    "roles/ml.developer"
  ])
  
  project = local.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.anomaly_detector.email}"
}

resource "google_project_iam_member" "healing_engine_permissions" {
  for_each = toset([
    "roles/bigquery.dataEditor",
    "roles/bigquery.jobUser",
    "roles/pubsub.publisher",
    "roles/pubsub.subscriber",
    "roles/datastore.user",
    "roles/cloudfunctions.invoker",
    "roles/run.invoker"
  ])
  
  project = local.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.healing_engine.email}"
}

resource "google_project_iam_member" "red_team_pentester_permissions" {
  for_each = toset([
    "roles/bigquery.dataEditor",
    "roles/bigquery.jobUser",
    "roles/pubsub.publisher",
    "roles/pubsub.subscriber",
    "roles/datastore.user",
    "roles/secretmanager.secretAccessor",
    "roles/logging.logWriter",
    "roles/aiplatform.user"
  ])
  
  project = local.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.red_team_pentester.email}"
}

resource "google_project_iam_member" "ado_orchestrator_permissions" {
  for_each = local.deploy_phase_2 ? toset([
    "roles/bigquery.dataEditor",
    "roles/bigquery.jobUser",
    "roles/pubsub.publisher",
    "roles/pubsub.subscriber",
    "roles/datastore.user",
    "roles/secretmanager.secretAccessor",
    "roles/aiplatform.user"
  ]) : toset([])
  
  project = local.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.ado_orchestrator[0].email}"
}

# Pub/Sub topics for agent communication
resource "google_pubsub_topic" "pipeline_events" {
  name = "aetherveil-pipeline-events"
  
  message_storage_policy {
    allowed_persistence_regions = [local.region]
  }
  
  schema_settings {
    schema   = google_pubsub_schema.pipeline_event.id
    encoding = "JSON"
  }
}

resource "google_pubsub_schema" "pipeline_event" {
  name       = "pipeline-event-schema"
  type       = "AVRO"
  definition = file("${path.module}/schemas/pipeline_event.avsc")
}

resource "google_pubsub_topic" "anomaly_alerts" {
  name = "aetherveil-anomaly-alerts"
  
  message_storage_policy {
    allowed_persistence_regions = [local.region]
  }
}

resource "google_pubsub_topic" "performance_metrics" {
  name = "aetherveil-performance-metrics"
  
  message_storage_policy {
    allowed_persistence_regions = [local.region]
  }
}

resource "google_pubsub_topic" "healing_actions" {
  name = "aetherveil-healing-actions"
  
  message_storage_policy {
    allowed_persistence_regions = [local.region]
  }
}

resource "google_pubsub_topic" "security_events" {
  name = "aetherveil-security-events"
  
  message_storage_policy {
    allowed_persistence_regions = [local.region]
  }
}

resource "google_pubsub_topic" "security_findings" {
  name = "aetherveil-security-findings"
  
  message_storage_policy {
    allowed_persistence_regions = [local.region]
  }
}

resource "google_pubsub_topic" "security_alerts" {
  name = "aetherveil-security-alerts"
  
  message_storage_policy {
    allowed_persistence_regions = [local.region]
  }
}

# Subscriptions for each agent
resource "google_pubsub_subscription" "pipeline_observer_sub" {
  name  = "pipeline-observer-subscription"
  topic = google_pubsub_topic.pipeline_events.name
  
  ack_deadline_seconds = 300
  
  retry_policy {
    minimum_backoff = "10s"
    maximum_backoff = "600s"
  }
  
  dead_letter_policy {
    dead_letter_topic     = google_pubsub_topic.dead_letter.id
    max_delivery_attempts = 5
  }
}

resource "google_pubsub_topic" "dead_letter" {
  name = "aetherveil-dead-letter"
}

# Cost-optimized BigQuery datasets
resource "google_bigquery_dataset" "pipeline_analytics" {
  dataset_id                      = "pipeline_analytics"
  location                        = "US"
  description                     = "Pipeline performance and anomaly analytics"
  delete_contents_on_destroy     = true
  default_table_expiration_ms    = 2592000000  # 30 days auto-deletion
  
  # Remove encryption to save KMS costs for non-sensitive data
  # default_encryption_configuration {
  #   kms_key_name = google_kms_crypto_key.main.id
  # }
}

# Combine ML and security datasets to save costs
resource "google_bigquery_dataset" "security_analytics" {
  dataset_id                      = "security_analytics"
  location                        = "US"
  description                     = "Security and ML analytics (combined for cost savings)"
  delete_contents_on_destroy     = true
  default_table_expiration_ms    = 2592000000  # 30 days auto-deletion
  
  # Keep encryption only for security data
  default_encryption_configuration {
    kms_key_name = google_kms_crypto_key.main.id
  }
}

# BigQuery tables for pipeline data
resource "google_bigquery_table" "pipeline_runs" {
  dataset_id = google_bigquery_dataset.pipeline_analytics.dataset_id
  table_id   = "pipeline_runs"
  
  schema = file("${path.module}/schemas/pipeline_runs.json")
  
  time_partitioning {
    type  = "DAY"
    field = "timestamp"
  }
  
  clustering = ["workflow_name", "status", "environment"]
}

resource "google_bigquery_table" "build_metrics" {
  dataset_id = google_bigquery_dataset.pipeline_analytics.dataset_id
  table_id   = "build_metrics"
  
  schema = file("${path.module}/schemas/build_metrics.json")
  
  time_partitioning {
    type  = "DAY"
    field = "timestamp"
  }
  
  clustering = ["component", "branch", "commit_sha"]
}

# Firestore for operational data storage
resource "google_firestore_database" "main" {
  project                     = local.project_id
  name                       = "(default)"
  location_id                = local.region
  type                       = "FIRESTORE_NATIVE"
  concurrency_mode           = "OPTIMISTIC"
  app_engine_integration_mode = "DISABLED"
}

# Firestore security rules
resource "google_firestore_document" "security_rules" {
  project     = local.project_id
  collection  = "_config"
  document_id = "security_rules"
  fields = jsonencode({
    rules = {
      stringValue = <<-EOT
        rules_version = '2';
        service cloud.firestore {
          match /databases/{database}/documents {
            // Agent state collections
            match /agent_state/{agentId} {
              allow read, write: if request.auth != null 
                && request.auth.token.email.matches('.*@${local.project_id}.iam.gserviceaccount.com');
            }
            
            // Healing history
            match /healing_history/{eventId} {
              allow read, write: if request.auth != null
                && request.auth.token.email.matches('.*@${local.project_id}.iam.gserviceaccount.com');
            }
            
            // Configuration data
            match /config/{configId} {
              allow read: if request.auth != null;
              allow write: if request.auth != null 
                && request.auth.token.email.matches('.*@${local.project_id}.iam.gserviceaccount.com');
            }
            
            // ML model metadata
            match /ml_models/{modelId} {
              allow read, write: if request.auth != null
                && request.auth.token.email.matches('.*@${local.project_id}.iam.gserviceaccount.com');
            }
            
            // Orchestration plans
            match /orchestration_plans/{planId} {
              allow read, write: if request.auth != null
                && request.auth.token.email.matches('.*@${local.project_id}.iam.gserviceaccount.com');
            }
          }
        }
      EOT
    }
  })
  
  depends_on = [google_firestore_database.main]
}

# VPC for secure communication
resource "google_compute_network" "vpc" {
  name                    = "aetherveil-vpc"
  auto_create_subnetworks = false
}

resource "google_compute_subnetwork" "main" {
  name          = "aetherveil-subnet"
  ip_cidr_range = "10.0.0.0/16"
  region        = local.region
  network       = google_compute_network.vpc.id
  
  private_ip_google_access = true
}

# Cloud Run services for Phase 1 agents
resource "google_cloud_run_service" "pipeline_observer" {
  count    = local.deploy_phase_1 ? 1 : 0
  name     = "pipeline-observer"
  location = local.region
  
  template {
    spec {
      service_account_name = google_service_account.pipeline_observer.email
      
      containers {
        image = "us-central1-docker.pkg.dev/${local.project_id}/aetherveil/pipeline-observer:latest"
        
        env {
          name  = "PROJECT_ID"
          value = local.project_id
        }
        
        env {
          name  = "PUBSUB_TOPIC"
          value = google_pubsub_topic.pipeline_events.name
        }
        
        env {
          name  = "BIGQUERY_DATASET"
          value = google_bigquery_dataset.pipeline_analytics.dataset_id
        }
        
        env {
          name  = "SPOOFING_ENABLED"
          value = var.enable_spoofing ? "true" : "false"
        }
        
        env {
          name  = "SUPPORTED_REGIONS"
          value = join(",", local.all_regions)
        }
        
        env {
          name  = "REGION_MAPPING"
          value = jsonencode(local.region_mapping)
        }
        
        resources {
          limits = {
            cpu    = "1000m"  # Reduced from 2000m
            memory = "1Gi"    # Reduced from 4Gi
          }
          requests = {
            cpu    = "100m"   # Minimal baseline
            memory = "256Mi" 
          }
        }
      }
    }
    
    metadata {
      annotations = {
        "autoscaling.knative.dev/minScale" = "0"  # Scale to zero for cost savings
        "autoscaling.knative.dev/maxScale" = "3"   # Reduced max scale
        "run.googleapis.com/cpu-throttling" = "true"
        "run.googleapis.com/execution-environment" = "gen2"
        # Remove VPC connector to save costs - use serverless VPC access instead
      }
    }
  }
  
  traffic {
    percent         = 100
    latest_revision = true
  }
}

# Red Team Pentester Agent (Phase 2)
module "red_team_pentester_v2" {
  count = local.deploy_phase_2 ? 1 : 0
  source = "./modules/red_team_agent_v2"

  project_id                   = local.project_id
  region                       = local.region
  service_account_email        = google_service_account.red_team_pentester.email
  artifact_registry_repository = "aetherveil"
  bigquery_dataset_id          = google_bigquery_dataset.security_analytics.dataset_id
  vpc_connector_id             = null  # VPC connector removed for cost optimization
  # hackerone_scope_file_path = "gs://your-bucket/hackerone_scope.json" # Uncomment and configure if using GCS
}

# Cost optimization: Remove VPC connector, use serverless VPC access
# Saves ~$50-100/month
# resource "google_vpc_access_connector" "main" {
#   name          = "aetherveil-connector"
#   region        = local.region
#   ip_cidr_range = "10.8.0.0/28"
#   network       = google_compute_network.vpc.name
# }

# Cloud Functions for event processing
resource "google_cloudfunctions2_function" "webhook_processor" {
  count    = local.deploy_phase_1 ? 1 : 0
  name     = "github-webhook-processor"
  location = local.region
  
  build_config {
    runtime     = "python311"
    entry_point = "process_webhook"
    
    source {
      storage_source {
        bucket = google_storage_bucket.function_source.name
        object = google_storage_bucket_object.webhook_processor_source.name
      }
    }
  }
  
  service_config {
    max_instance_count = 100
    min_instance_count = 1
    available_memory   = "256M"
    timeout_seconds    = 60
    
    environment_variables = {
      PROJECT_ID    = local.project_id
      PUBSUB_TOPIC  = google_pubsub_topic.pipeline_events.name
    }
    
    service_account_email = google_service_account.pipeline_observer.email
  }
  
  event_trigger {
    trigger_region = local.region
    event_type     = "google.cloud.pubsub.topic.v1.messagePublished"
    pubsub_topic   = google_pubsub_topic.pipeline_events.id
  }
}

# Cost-optimized storage for Cloud Functions
resource "google_storage_bucket" "function_source" {
  name          = "${local.project_id}-functions"
  location      = local.region
  storage_class = "STANDARD"  # Standard is often cheapest for small files
  force_destroy = true
  
  # Remove encryption to save KMS costs
  # encryption {
  #   default_kms_key_name = google_kms_crypto_key.main.id
  # }
  
  lifecycle_rule {
    condition {
      age = 7  # Clean up old function sources after 7 days
    }
    action {
      type = "Delete"
    }
  }
}

# Placeholder - actual source will be uploaded during deployment
resource "google_storage_bucket_object" "webhook_processor_source" {
  name    = "webhook-processor-source.zip"
  bucket  = google_storage_bucket.function_source.name
  content = "placeholder"  # Will be replaced during deployment
}

# Vertex AI for ML models (Phase 2)
resource "google_vertex_ai_endpoint" "anomaly_detection" {
  count        = local.deploy_phase_2 ? 1 : 0
  name         = "anomaly-detection-endpoint"
  display_name = "Pipeline Anomaly Detection"
  location     = local.region
  
  encryption_spec {
    kms_key_name = google_kms_crypto_key.main.id
  }
}

# Cost optimization: Remove GKE cluster for Phase 1 (not needed)
# Saves ~$200-500/month
# GKE cluster will be added in Phase 2 when budget increases
# 
# resource "google_container_cluster" "ado_cluster" {
#   count    = local.deploy_phase_2 ? 1 : 0
#   name     = "ado-cluster"
#   location = local.region
#   ...
# }

# Monitoring and alerting
resource "google_monitoring_alert_policy" "pipeline_failure" {
  display_name = "Pipeline Failure Alert"
  combiner     = "OR"
  
  conditions {
    display_name = "Pipeline failure rate"
    
    condition_threshold {
      filter          = "resource.type=\"cloud_run_revision\" AND metric.type=\"run.googleapis.com/request_count\""
      duration        = "300s"
      comparison      = "COMPARISON_GT"
      threshold_value = 0.1
      
      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_RATE"
      }
    }
  }
  
  notification_channels = [google_monitoring_notification_channel.email.name]
}

resource "google_monitoring_notification_channel" "email" {
  display_name = "Email Notification"
  type         = "email"
  
  labels = {
    email_address = "alerts@aetherveil.com"
  }
}

# Outputs
output "pipeline_observer_url" {
  value = local.deploy_phase_1 ? google_cloud_run_service.pipeline_observer[0].status[0].url : null
}

output "bigquery_dataset" {
  value = google_bigquery_dataset.pipeline_analytics.dataset_id
}

output "firestore_database" {
  value = google_firestore_database.main.name
}

output "pubsub_topics" {
  value = {
    pipeline_events    = google_pubsub_topic.pipeline_events.name
    anomaly_alerts     = google_pubsub_topic.anomaly_alerts.name
    performance_metrics = google_pubsub_topic.performance_metrics.name
    healing_actions    = google_pubsub_topic.healing_actions.name
    security_events    = google_pubsub_topic.security_events.name
  }
}

output "service_accounts" {
  value = {
    pipeline_observer = google_service_account.pipeline_observer.email
    anomaly_detector  = google_service_account.anomaly_detector.email
    healing_engine    = google_service_account.healing_engine.email
    ado_orchestrator  = local.deploy_phase_2 ? google_service_account.ado_orchestrator[0].email : null
  }
}

# === SPOOFING OUTPUTS ===
output "spoofing_info" {
  value = var.enable_spoofing ? {
    global_ip              = google_compute_global_address.aetherveil_global[0].address
    dns_zone               = google_dns_managed_zone.aetherveil_zone[0].dns_name
    regional_endpoints     = [for region in local.all_regions : "${region}.${var.domain_name}"]
    cdn_enabled           = true
    dummy_regions         = local.spoofed_regions
    ssl_certificate_domains = google_compute_managed_ssl_certificate.aetherveil_cert[0].managed[0].domains
  } : null
}

output "verification_commands" {
  value = var.enable_spoofing ? {
    check_global_ip     = "dig ${var.domain_name}"
    check_regional_dns  = [for region in local.all_regions : "dig ${region}.${var.domain_name}"]
    test_latency_spoofing = [for region in local.all_regions : "curl -H 'Host: ${region}.${var.domain_name}' https://${google_compute_global_address.aetherveil_global[0].address}/ -I"]
    check_cdn_headers   = "curl -H 'Host: ${var.domain_name}' https://${google_compute_global_address.aetherveil_global[0].address}/ -I"
  } : null
}

# Synthetic audit logging for fake regional activity
resource "google_bigquery_table" "synthetic_audit_logs" {
  count      = var.enable_spoofing ? 1 : 0
  dataset_id = google_bigquery_dataset.security_analytics.dataset_id
  table_id   = "synthetic_audit_logs"
  
  schema = jsonencode([
    {
      name = "timestamp"
      type = "TIMESTAMP"
      mode = "REQUIRED"
    },
    {
      name = "region"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "service"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "operation"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "resource_name"
      type = "STRING"
      mode = "NULLABLE"
    },
    {
      name = "principal_email"
      type = "STRING"
      mode = "NULLABLE"
    },
    {
      name = "source_ip"
      type = "STRING"
      mode = "NULLABLE"
    },
    {
      name = "user_agent"
      type = "STRING"
      mode = "NULLABLE"
    }
  ])
  
  time_partitioning {
    type  = "DAY"
    field = "timestamp"
  }
  
  clustering = ["region", "service", "operation"]
}

# Cost-optimized: Reduce synthetic log frequency
resource "google_cloud_scheduler_job" "synthetic_audit_generator" {
  count            = var.enable_spoofing ? 1 : 0
  name             = "synthetic-audit-generator"
  description      = "Generate synthetic audit logs for regional spoofing"
  schedule         = "0 */6 * * *"  # Every 6 hours instead of 15 minutes
  time_zone        = "UTC"
  region           = local.region
  
  http_target {
    http_method = "POST"
    uri         = google_cloud_run_service.pipeline_observer[0].status[0].url
    
    headers = {
      "Content-Type" = "application/json"
    }
    
    body = base64encode(jsonencode({
      action = "generate_synthetic_audit_logs"
      regions = local.spoofed_regions
    }))
    
    oidc_token {
      service_account_email = google_service_account.pipeline_observer.email
    }
  }
}