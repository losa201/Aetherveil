# APOA - Initial Implementation Plan

This document outlines the steps, resources, and technologies required for the initial implementation (Phase 1) of the Autonomous Pipeline Observer Agent (APOA).

## 1. Technology Stack

- **Programming Language:** Python 3.11+
- **Web Framework:** FastAPI (with Uvicorn) for its high performance, async capabilities, and automatic data validation via Pydantic.
- **Core Libraries:**
    - `google-cloud-pubsub`: For interacting with Pub/Sub topics.
    - `google-cloud-bigquery`: For streaming data to BigQuery.
    - `google-cloud-aiplatform`: For making predictions against Vertex AI endpoints.
    - `pydantic`: For data modeling and validation.
- **Containerization:** Docker

## 2. GCP Resources to Provision

The following GCP resources need to be created and configured:

1.  **IAM Service Account (SA):**
    - **Name:** `apoa-agent-sa`
    - **Roles:**
        - `roles/pubsub.subscriber` (on ingestion topics)
        - `roles/pubsub.publisher` (on findings topic)
        - `roles/bigquery.dataEditor` (on the `pipeline_analytics` dataset)
        - `roles/aiplatform.user` (to invoke Vertex AI models)
        - `roles/logging.logWriter` (for application logging)
    - This SA will be used as the identity for the Cloud Run service.

2.  **Pub/Sub Topics:**
    - `github-events`: To receive raw webhook data from GitHub.
    - `pipeline-logs`: To receive logs from the Cloud Logging sink.
    - `pipeline-metrics`: To receive metrics from the Cloud Monitoring sink.
    - `apoa-findings`: To publish insights and actions.

3.  **Cloud Logging Sink:**
    - **Source:** Logs from services related to CI/CD (e.g., Cloud Build, GKE).
    - **Filter:** `resource.type="cloud_build_step" OR resource.type="k8s_container"` (example filter, needs refinement).
    - **Destination:** The `pipeline-logs` Pub/Sub topic.

4.  **BigQuery Dataset:**
    - **Name:** `pipeline_analytics`
    - **Tables:**
        - `workflow_runs`: Schema to store normalized data from GitHub workflow events (run ID, job name, duration, status, etc.).
        - `log_entries`: Schema for structured log data.

5.  **Cloud Run Service:**
    - **Service Name:** `apoa-agent`
    - **Container Image:** To be built from the APOA agent's Dockerfile.
    - **Service Account:** `apoa-agent-sa`
    - **Triggers:**
        - Push subscription to the `github-events` topic.
        - Push subscription to the `pipeline-logs` topic.
        - Push subscription to the `pipeline-metrics` topic.
    - **Environment Variables:**
        - `GCP_PROJECT_ID`: The GCP Project ID.
        - `BIGQUERY_DATASET`: `pipeline_analytics`
        - `FINDINGS_TOPIC`: `apoa-findings`
        - `VERTEX_AI_ENDPOINT`: The full resource name of the Vertex AI prediction endpoint.

## 3. Agent Application Structure (MVP)

A new directory `agents/pipeline_observer/` will be created with the following structure:

```
agents/pipeline_observer/
├── __init__.py
├── main.py             # FastAPI app, endpoints, Pub/Sub subscriber logic
├── schemas.py          # Pydantic models for incoming data
├── services.py         # Business logic (processing, analysis)
├── requirements.txt    # Python dependencies
└── Dockerfile          # Container definition
```

### Example API Endpoints

The service will be primarily event-driven via Pub/Sub push subscriptions, but will expose a few HTTP endpoints:

- **`POST /v1/subscriber/github-events`**: The push endpoint for the `github-events` topic.
- **`POST /v1/subscriber/logs`**: The push endpoint for the `pipeline-logs` topic.
- **`GET /health`**: A simple health check endpoint that returns `{"status": "ok"}`.

## 4. Implementation Steps (Phase 1 MVP)

1.  **Scaffold GCP Resources:** Use Terraform or `gcloud` CLI commands to provision the IAM, Pub/Sub, and BigQuery resources defined above.
2.  **Develop Agent Skeleton:**
    - Create the directory structure under `agents/`.
    - Initialize the FastAPI application in `main.py`.
    - Implement the `/health` endpoint.
    - Create Pydantic schemas in `schemas.py` for a Pub/Sub push request and a simple GitHub workflow event.
3.  **Implement Pub/Sub Subscriber Logic:**
    - In `main.py`, create the POST endpoints for the push subscriptions.
    - The endpoint logic will:
        a. Receive the request and decode the Pub/Sub message data using the Pydantic schema.
        b. (MVP) Log the decoded data to standard output.
        c. (Full) Pass the data to a processing function in `services.py`.
4.  **Containerize the Application:**
    - Write a `Dockerfile` that uses a Python base image, installs dependencies from `requirements.txt`, and runs the application using `uvicorn`.
5.  **Deploy to Cloud Run:**
    - Build the Docker image and push it to Google Artifact Registry.
    - Deploy the image as a new Cloud Run service, configuring it with the appropriate service account and environment variables.
    - Configure the Pub/Sub push subscriptions to target the deployed service's subscriber endpoints.
6.  **Setup GitHub Webhook:**
    - Create a simple, publicly accessible Cloud Function that acts as a webhook receiver.
    - This function's only job is to validate the GitHub webhook signature (for security) and publish the payload to the `github-events` Pub/Sub topic.
    - Configure the GitHub repository to send `workflow_run` webhooks to this Cloud Function's URL.
