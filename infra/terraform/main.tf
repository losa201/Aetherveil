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

locals {
  project_id = var.project_id
  region     = var.region
  zone       = var.zone
  
  # Phase deployment flags
  deploy_phase_1 = var.deploy_phase_1
  deploy_phase_2 = var.deploy_phase_2
  deploy_phase_3 = var.deploy_phase_3
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
    "binaryauthorization.googleapis.com"
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
    "roles/logging.logWriter"
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

# BigQuery datasets for analytics
resource "google_bigquery_dataset" "pipeline_analytics" {
  dataset_id  = "pipeline_analytics"
  location    = "US"
  description = "Pipeline performance and anomaly analytics"
  
  default_encryption_configuration {
    kms_key_name = google_kms_crypto_key.main.id
  }
}

resource "google_bigquery_dataset" "ml_models" {
  dataset_id  = "ml_models"
  location    = "US"
  description = "Machine learning models for DevOps automation"
  
  default_encryption_configuration {
    kms_key_name = google_kms_crypto_key.main.id
  }
}

resource "google_bigquery_dataset" "security_analytics" {
  dataset_id  = "security_analytics"
  location    = "US"
  description = "Security vulnerability findings and analytics"
  
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
        
        resources {
          limits = {
            cpu    = "2000m"
            memory = "4Gi"
          }
        }
      }
    }
    
    metadata {
      annotations = {
        "autoscaling.knative.dev/minScale" = "1"
        "autoscaling.knative.dev/maxScale" = "10"
        "run.googleapis.com/vpc-access-connector" = google_vpc_access_connector.main.id
      }
    }
  }
  
  traffic {
    percent         = 100
    latest_revision = true
  }
}

# VPC Connector for Cloud Run
resource "google_vpc_access_connector" "main" {
  name          = "aetherveil-connector"
  region        = local.region
  ip_cidr_range = "10.8.0.0/28"
  network       = google_compute_network.vpc.name
}

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

# Storage for Cloud Functions source
resource "google_storage_bucket" "function_source" {
  name     = "${local.project_id}-function-source"
  location = local.region
  
  encryption {
    default_kms_key_name = google_kms_crypto_key.main.id
  }
}

resource "google_storage_bucket_object" "webhook_processor_source" {
  name   = "webhook-processor-source.zip"
  bucket = google_storage_bucket.function_source.name
  source = "${path.module}/functions/webhook_processor.zip"
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

# GKE cluster for Phase 2/3 advanced agents
resource "google_container_cluster" "ado_cluster" {
  count    = local.deploy_phase_2 ? 1 : 0
  name     = "ado-cluster"
  location = local.region
  
  remove_default_node_pool = true
  initial_node_count       = 1
  
  network    = google_compute_network.vpc.name
  subnetwork = google_compute_subnetwork.main.name
  
  workload_identity_config {
    workload_pool = "${local.project_id}.svc.id.goog"
  }
  
  database_encryption {
    state    = "ENCRYPTED"
    key_name = google_kms_crypto_key.main.id
  }
  
  binary_authorization {
    evaluation_mode = "PROJECT_SINGLETON_POLICY_ENFORCE"
  }
}

resource "google_container_node_pool" "ado_nodes" {
  count      = local.deploy_phase_2 ? 1 : 0
  name       = "ado-node-pool"
  location   = local.region
  cluster    = google_container_cluster.ado_cluster[0].name
  node_count = 3
  
  node_config {
    preemptible  = false
    machine_type = "e2-standard-4"
    
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
    
    workload_metadata_config {
      mode = "GKE_METADATA"
    }
    
    shielded_instance_config {
      enable_secure_boot          = true
      enable_integrity_monitoring = true
    }
  }
  
  management {
    auto_repair  = true
    auto_upgrade = true
  }
}

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