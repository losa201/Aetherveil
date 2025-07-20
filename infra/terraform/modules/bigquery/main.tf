# BigQuery Module for Aetherveil AI Pentesting Data

terraform {
  required_version = ">= 1.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 4.84"
    }
  }
}

# BigQuery Dataset
resource "google_bigquery_dataset" "aetherveil_dataset" {
  dataset_id    = var.dataset_id
  friendly_name = "Aetherveil AI Pentesting Data"
  description   = "Dataset containing pentesting results, learning data, and performance metrics"
  location      = var.region
  project       = var.project_id

  # Data retention and lifecycle
  default_table_expiration_ms = var.table_expiration_days * 24 * 60 * 60 * 1000
  delete_contents_on_destroy  = var.delete_contents_on_destroy

  # Access control
  access {
    role          = "OWNER"
    user_by_email = var.admin_email
  }

  access {
    role           = "READER"
    special_group  = "projectReaders"
  }

  access {
    role           = "WRITER"
    special_group  = "projectWriters"
  }

  labels = {
    environment = var.environment
    project     = "aetherveil"
    component   = "bigquery"
  }
}

# Pentesting Results Table
resource "google_bigquery_table" "pentesting_results" {
  dataset_id = google_bigquery_dataset.aetherveil_dataset.dataset_id
  table_id   = var.results_table_id
  project    = var.project_id

  description = "Pentesting execution results and findings"

  time_partitioning {
    type                     = "DAY"
    field                    = "timestamp"
    require_partition_filter = false
    expiration_ms           = var.partition_expiration_days * 24 * 60 * 60 * 1000
  }

  clustering = ["category", "tool", "severity"]

  schema = jsonencode([
    {
      name = "timestamp"
      type = "TIMESTAMP"
      mode = "REQUIRED"
      description = "Execution timestamp"
    },
    {
      name = "session_id"
      type = "STRING"
      mode = "REQUIRED"
      description = "Unique session identifier"
    },
    {
      name = "task_id"
      type = "STRING"
      mode = "REQUIRED"
      description = "Unique task identifier"
    },
    {
      name = "tool"
      type = "STRING"
      mode = "REQUIRED"
      description = "Tool used for testing"
    },
    {
      name = "category"
      type = "STRING"
      mode = "REQUIRED"
      description = "Testing category (web, api, cloud, etc.)"
    },
    {
      name = "target"
      type = "STRING"
      mode = "NULLABLE"
      description = "Target being tested (hashed for privacy)"
    },
    {
      name = "status"
      type = "STRING"
      mode = "REQUIRED"
      description = "Task execution status"
    },
    {
      name = "duration"
      type = "FLOAT"
      mode = "NULLABLE"
      description = "Execution duration in seconds"
    },
    {
      name = "findings_count"
      type = "INTEGER"
      mode = "NULLABLE"
      description = "Number of findings discovered"
    },
    {
      name = "vulnerabilities_count"
      type = "INTEGER"
      mode = "NULLABLE"
      description = "Number of vulnerabilities found"
    },
    {
      name = "severity"
      type = "STRING"
      mode = "NULLABLE"
      description = "Highest severity level found"
    },
    {
      name = "risk_assessment"
      type = "STRING"
      mode = "NULLABLE"
      description = "Overall risk assessment"
    },
    {
      name = "vulnerabilities"
      type = "JSON"
      mode = "NULLABLE"
      description = "Detailed vulnerability information"
    },
    {
      name = "findings"
      type = "JSON"
      mode = "NULLABLE"
      description = "Detailed findings information"
    },
    {
      name = "execution_metadata"
      type = "JSON"
      mode = "NULLABLE"
      description = "Execution metadata and context"
    },
    {
      name = "agent_version"
      type = "STRING"
      mode = "NULLABLE"
      description = "Agent version that generated the result"
    },
    {
      name = "created_at"
      type = "TIMESTAMP"
      mode = "REQUIRED"
      description = "Record creation timestamp"
    }
  ])

  labels = {
    environment = var.environment
    project     = "aetherveil"
    table_type  = "results"
  }
}

# Learning Data Table
resource "google_bigquery_table" "learning_data" {
  dataset_id = google_bigquery_dataset.aetherveil_dataset.dataset_id
  table_id   = var.learning_table_id
  project    = var.project_id

  description = "AI learning patterns and model improvement data"

  time_partitioning {
    type                     = "DAY"
    field                    = "timestamp"
    require_partition_filter = false
    expiration_ms           = var.partition_expiration_days * 24 * 60 * 60 * 1000
  }

  clustering = ["category", "pattern_type", "effectiveness_score"]

  schema = jsonencode([
    {
      name = "timestamp"
      type = "TIMESTAMP"
      mode = "REQUIRED"
      description = "Learning event timestamp"
    },
    {
      name = "session_id"
      type = "STRING"
      mode = "REQUIRED"
      description = "Session that generated the learning"
    },
    {
      name = "category"
      type = "STRING"
      mode = "REQUIRED"
      description = "Testing category"
    },
    {
      name = "pattern_type"
      type = "STRING"
      mode = "REQUIRED"
      description = "Type of learned pattern"
    },
    {
      name = "pattern_data"
      type = "JSON"
      mode = "REQUIRED"
      description = "Detailed pattern information"
    },
    {
      name = "effectiveness_score"
      type = "FLOAT"
      mode = "NULLABLE"
      description = "Pattern effectiveness score (0-1)"
    },
    {
      name = "confidence"
      type = "FLOAT"
      mode = "NULLABLE"
      description = "Confidence in the pattern (0-1)"
    },
    {
      name = "success_count"
      type = "INTEGER"
      mode = "NULLABLE"
      description = "Number of successful applications"
    },
    {
      name = "failure_count"
      type = "INTEGER"
      mode = "NULLABLE"
      description = "Number of failed applications"
    },
    {
      name = "context"
      type = "JSON"
      mode = "NULLABLE"
      description = "Learning context and metadata"
    },
    {
      name = "agent_version"
      type = "STRING"
      mode = "NULLABLE"
      description = "Agent version that generated the learning"
    },
    {
      name = "created_at"
      type = "TIMESTAMP"
      mode = "REQUIRED"
      description = "Record creation timestamp"
    }
  ])

  labels = {
    environment = var.environment
    project     = "aetherveil"
    table_type  = "learning"
  }
}

# Performance Metrics Table
resource "google_bigquery_table" "performance_metrics" {
  dataset_id = google_bigquery_dataset.aetherveil_dataset.dataset_id
  table_id   = var.metrics_table_id
  project    = var.project_id

  description = "Agent performance metrics and system statistics"

  time_partitioning {
    type                     = "DAY"
    field                    = "timestamp"
    require_partition_filter = false
    expiration_ms           = var.partition_expiration_days * 24 * 60 * 60 * 1000
  }

  clustering = ["metric_category", "agent_id"]

  schema = jsonencode([
    {
      name = "timestamp"
      type = "TIMESTAMP"
      mode = "REQUIRED"
      description = "Metric collection timestamp"
    },
    {
      name = "agent_id"
      type = "STRING"
      mode = "REQUIRED"
      description = "Unique agent instance identifier"
    },
    {
      name = "session_id"
      type = "STRING"
      mode = "NULLABLE"
      description = "Current session identifier"
    },
    {
      name = "metric_category"
      type = "STRING"
      mode = "REQUIRED"
      description = "Category of metric (performance, resource, etc.)"
    },
    {
      name = "metric_name"
      type = "STRING"
      mode = "REQUIRED"
      description = "Specific metric name"
    },
    {
      name = "metric_value"
      type = "FLOAT"
      mode = "REQUIRED"
      description = "Metric value"
    },
    {
      name = "metric_unit"
      type = "STRING"
      mode = "NULLABLE"
      description = "Unit of measurement"
    },
    {
      name = "system_info"
      type = "JSON"
      mode = "NULLABLE"
      description = "System information and context"
    },
    {
      name = "agent_version"
      type = "STRING"
      mode = "NULLABLE"
      description = "Agent version"
    },
    {
      name = "created_at"
      type = "TIMESTAMP"
      mode = "REQUIRED"
      description = "Record creation timestamp"
    }
  ])

  labels = {
    environment = var.environment
    project     = "aetherveil"
    table_type  = "metrics"
  }
}

# Scheduled Queries for Analytics
resource "google_bigquery_table" "vulnerability_trends" {
  dataset_id = google_bigquery_dataset.aetherveil_dataset.dataset_id
  table_id   = "vulnerability_trends"
  project    = var.project_id

  description = "Materialized view of vulnerability trends over time"

  materialized_view {
    query = <<EOF
SELECT
  DATE(timestamp) as date,
  category,
  COUNT(*) as total_scans,
  SUM(vulnerabilities_count) as total_vulnerabilities,
  AVG(vulnerabilities_count) as avg_vulnerabilities_per_scan,
  COUNT(DISTINCT session_id) as unique_sessions,
  ARRAY_AGG(DISTINCT tool) as tools_used
FROM `${var.project_id}.${var.dataset_id}.${var.results_table_id}`
WHERE timestamp >= DATETIME_SUB(CURRENT_DATETIME(), INTERVAL 30 DAY)
  AND status = 'completed'
GROUP BY date, category
ORDER BY date DESC, category
EOF
  }

  labels = {
    environment = var.environment
    project     = "aetherveil"
    table_type  = "analytics"
  }
}

# Data Transfer Configuration for ML
resource "google_bigquery_data_transfer_config" "ml_training_export" {
  count = var.enable_ml_pipeline ? 1 : 0

  display_name   = "Aetherveil ML Training Data Export"
  location       = var.region
  data_source_id = "scheduled_query"
  
  destination_dataset_id = google_bigquery_dataset.aetherveil_dataset.dataset_id
  
  schedule = "every 24 hours"
  
  params = {
    destination_table_name_template = "ml_training_data_{run_date}"
    write_disposition              = "WRITE_TRUNCATE"
    query = <<EOF
SELECT
  category,
  tool,
  target,
  status,
  duration,
  vulnerabilities_count,
  risk_assessment,
  JSON_EXTRACT_SCALAR(execution_metadata, '$.success_rate') as success_rate,
  JSON_EXTRACT_SCALAR(execution_metadata, '$.effectiveness_score') as effectiveness_score
FROM `${var.project_id}.${var.dataset_id}.${var.results_table_id}`
WHERE timestamp >= DATETIME_SUB(CURRENT_DATETIME(), INTERVAL 7 DAY)
  AND status = 'completed'
  AND vulnerabilities_count IS NOT NULL
EOF
  }

  labels = {
    environment = var.environment
    project     = "aetherveil"
    component   = "ml-pipeline"
  }
}

# BigQuery ML Model for Vulnerability Prediction
resource "google_bigquery_table" "vulnerability_prediction_model" {
  count = var.enable_ml_pipeline ? 1 : 0

  dataset_id = google_bigquery_dataset.aetherveil_dataset.dataset_id
  table_id   = "vulnerability_prediction_model"
  project    = var.project_id

  description = "ML model for predicting vulnerability likelihood"

  # This would typically be created via a scheduled query or manual ML training
  labels = {
    environment = var.environment
    project     = "aetherveil"
    table_type  = "ml-model"
  }
}

# IAM bindings for service account access
resource "google_bigquery_dataset_iam_member" "agent_access" {
  count = var.service_account_email != "" ? 1 : 0

  dataset_id = google_bigquery_dataset.aetherveil_dataset.dataset_id
  role       = "roles/bigquery.dataEditor"
  member     = "serviceAccount:${var.service_account_email}"
  project    = var.project_id
}