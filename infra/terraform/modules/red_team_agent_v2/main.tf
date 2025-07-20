resource "google_cloud_run_service" "red_team_pentester_v2" {
  name     = "red-team-pentester-v2"
  location = var.region
  project  = var.project_id

  template {
    spec {
      service_account_name = var.service_account_email
      containers {
        image = "${var.region}-docker.pkg.dev/${var.project_id}/${var.artifact_registry_repository}/red-team-pentester-v2:latest"
        env {
          name  = "PROJECT_ID"
          value = var.project_id
        }
        env {
          name  = "REGION"
          value = var.region
        }
        # Add other environment variables as needed by the agent
        # For example, BigQuery dataset ID, Secret Manager secret IDs
        env {
          name  = "BIGQUERY_DATASET_ID"
          value = var.bigquery_dataset_id
        }
        env {
          name  = "HACKERONE_SCOPE_FILE"
          value = var.hackerone_scope_file_path # Or fetch from Secret Manager
        }
        resources {
          limits = {
            cpu    = "1000m"
            memory = "2Gi"
          }
        }
      }
      timeout_seconds = 300 # 5 minutes for initial scan, adjust as needed
    }
    metadata {
      annotations = {
        "autoscaling.knative.dev/minScale" = "0"
        "autoscaling.knative.dev/maxScale" = "1"
        "run.googleapis.com/vpc-access-connector" = var.vpc_connector_id
      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }
}

# Output the URL of the deployed service
output "red_team_pentester_v2_url" {
  value = google_cloud_run_service.red_team_pentester_v2.status[0].url
}
