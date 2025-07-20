variable "project_id" {
  description = "The GCP project ID."
  type        = string
}

variable "region" {
  description = "The GCP region where the Cloud Run service will be deployed."
  type        = string
}

variable "service_account_email" {
  description = "The email of the service account to use for the Cloud Run service."
  type        = string
}

variable "artifact_registry_repository" {
  description = "The name of the Artifact Registry repository."
  type        = string
}

variable "bigquery_dataset_id" {
  description = "The ID of the BigQuery dataset for findings."
  type        = string
}

variable "hackerone_scope_file_path" {
  description = "Path to the HackerOne scope file (e.g., in GCS or a local path within the container)."
  type        = string
  default     = ""
}

variable "vpc_connector_id" {
  description = "The ID of the VPC Access Connector for private network access."
  type        = string
}
