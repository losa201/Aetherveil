# Aetherveil Autonomous AI DevOps Platform - Deployment Guide

This guide provides instructions for deploying the Aetherveil platform, including the AI Red Team Agent, to Google Cloud Platform (GCP).

## Prerequisites

Before you begin, ensure you have the following:

1.  **GCP Project**: A GCP project with billing enabled. (Project ID: `tidy-computing-465909-i3`)
2.  **GCP CLI**: `gcloud` CLI installed and authenticated. Ensure your authenticated user has sufficient permissions (e.g., Project Owner or roles equivalent to `roles/editor`, `roles/cloudbuild.builds.editor`, `roles/artifactregistry.writer`, `roles/run.admin`, `roles/iam.serviceAccountAdmin`, `roles/compute.networkAdmin`, `roles/bigquery.admin`, `roles/secretmanager.admin`, `roles/aiplatform.admin`).
3.  **Terraform**: Terraform CLI installed (version `~> 1.0`). The `deploy.sh` script will attempt to install it if not found.
4.  **Python**: Python 3.9+ and `pip` for running local scripts and tests.
5.  **Git**: Git installed for cloning the repository.

## Setup

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/your-repo/Aetherveil.git # Replace with actual repo URL
    cd Aetherveil
    ```

2.  **Configure GCP Project and Region**:
    Set your default GCP project and region. These values are used by the deployment scripts.
    ```bash
    gcloud config set project tidy-computing-465909-i3
    gcloud config set compute/region us-central1
    ```

3.  **Enable Required GCP APIs**:
    The `deploy.sh` script will attempt to enable these, but you can do it manually if preferred:
    ```bash
    gcloud services enable \
        run.googleapis.com \
        cloudbuild.googleapis.com \
        artifactregistry.googleapis.com \
        pubsub.googleapis.com \
        bigquery.googleapis.com \
        firestore.googleapis.com \
        aiplatform.googleapis.com \
        ml.googleapis.com \
        cloudscheduler.googleapis.com \
        cloudfunctions.googleapis.com \
        monitoring.googleapis.com \
        logging.googleapis.com \
        secretmanager.googleapis.com \
        container.googleapis.com \
        compute.googleapis.com \
        cloudkms.googleapis.com \
        securitycenter.googleapis.com \
        binaryauthorization.googleapis.com
    ```

4.  **Create Artifact Registry Repository**:
    The `deploy.sh` script will create this if it doesn't exist.
    ```bash
    gcloud artifacts repositories create aetherveil \
        --repository-format=docker \
        --location=us-central1 \
        --description="Aetherveil Autonomous DevOps Platform"
    ```

5.  **Setup Terraform GCS Backend**:
    The `deploy.sh` script will create the GCS bucket for Terraform state if it doesn't exist.
    ```bash
    gsutil mb gs://aetherveil-terraform-state-tidy-computing-465909-i3
    gsutil versioning set on gs://aetherveil-terraform-state-tidy-computing-465909-i3
    ```

## Deployment Steps

The deployment is managed by the `deploy.sh` script, which orchestrates Cloud Build for container images and Terraform for infrastructure.

**Important**: Ensure the service account `aetherveil-cicd@tidy-computing-465909-i3.iam.gserviceaccount.com` has the necessary permissions to run Terraform (e.g., `roles/editor` or more granular roles for specific resources).

### Deploying All Phases (Recommended for Full Functionality)

To deploy the entire Aetherveil platform, including the AI Red Team Agent (part of Phase 2):

```bash
./deploy.sh --phase 2
```

This command will:

1.  Check prerequisites and enable APIs.
2.  Create Artifact Registry and Terraform backend.
3.  Build and push all necessary container images (including `red-team-pentester-v2`) to Artifact Registry using Cloud Build.
4.  Deploy the core infrastructure (VPC, Pub/Sub, BigQuery, Firestore, KMS, Cloud Functions, Cloud Run services for Phase 1 agents) using Terraform.
5.  Deploy the AI Red Team Agent as a Cloud Run service.
6.  Set up BigQuery tables for findings and analytics.
7.  Configure basic monitoring and alerting.
8.  Run integration tests.

### Deploying Specific Phases

You can deploy specific phases if needed:

*   **Phase 1 (Core Platform)**:
    ```bash
    ./deploy.sh --phase 1
    ```
*   **Phase 3 (Future Expansion)**:
    ```bash
    ./deploy.sh --phase 3
    ```
    (Note: Phase 3 components are placeholders and will be implemented in future iterations.)

## Post-Deployment Configuration

1.  **HackerOne Scope**: The AI Red Team Agent needs a HackerOne scope file. For production, store this securely in GCP Secret Manager. Update the agent's environment variable `HACKERONE_SCOPE_FILE` in the Terraform module to point to a GCS bucket path or a Secret Manager secret ID.
    *   **Example (Secret Manager)**:
        ```bash
        echo '{"targets": [{"target": "https://example.com", "asset_type": "URL", "in_scope": true}]}' | \
        gcloud secrets create hackerone-scope --data-file=- --project=tidy-computing-465909-i3
        ```
        Then, modify `agents/red_team_pentester_v2/main.py` to fetch this secret.

2.  **Slack Webhook**: For reporting findings to Slack, create a Slack webhook and store it in Secret Manager:
    ```bash
    echo "YOUR_SLACK_WEBHOOK_URL" | \
    gcloud secrets create slack-red-team-webhook --data-file=- --project=tidy-computing-465909-i3
    ```

3.  **Cloud Function Webhook**: If you are integrating with GitHub or other SCMs, configure their webhooks to point to the deployed `github-webhook-processor` Cloud Function URL. You can find the URL in the Cloud Functions console.

## Verification

After deployment, verify the components:

*   **Cloud Run**: Check the Cloud Run services in the GCP Console. Ensure `red-team-pentester-v2` is deployed and running.
*   **BigQuery**: Verify the `security_analytics` dataset and `red_team_findings` table exist.
*   **Cloud Build**: Review recent builds in the Cloud Build history to ensure all agent images were built successfully.
*   **Logs**: Monitor Cloud Logging for agent activity and potential errors.

## Notes on Monitoring, Cost Optimization, and Agent Learning

### Monitoring

*   **Cloud Monitoring**: The platform deploys a basic dashboard and alert policies. Customize these in the GCP Console to fit your operational needs.
*   **Cloud Logging**: All agent logs are streamed to Cloud Logging. Use Log Explorer for detailed analysis, error debugging, and creating custom metrics.
*   **Cloud Trace**: For distributed tracing of requests across services (e.g., Cloud Run, Cloud Functions), enable Cloud Trace for deeper performance insights.

### Cost Optimization

*   **Cloud Run**: Configured with `minScale: 0` (except for `pipeline-observer` which is `minScale: 1` for continuous monitoring) to scale down to zero instances when idle, minimizing costs. Adjust `maxScale` based on expected load.
*   **BigQuery**: Cost-effective for large-scale analytics. Optimize queries and use partitioning/clustering to reduce scan costs.
*   **Cloud Functions**: Pay-per-invocation model. Optimize function execution time and memory.
*   **Vertex AI**: Monitor model training and prediction costs. Utilize managed datasets and smaller machine types for training where feasible.
*   **VPC Access Connector**: Costs are incurred for connector uptime. Ensure it's appropriately sized.
*   **KMS**: Minimal cost for key management, but consider key rotation policies.

### Agent Learning and Evolution

The AI Red Team Agent is designed for continuous learning and evolution:

*   **Findings as Data**: Every vulnerability finding stored in BigQuery serves as a data point for the agent's learning model.
*   **Vertex AI Integration**: The `learner.py` module is a placeholder for integrating with Vertex AI. In a full implementation, this would involve:
    *   **Data Preparation**: Extracting features from findings (e.g., vulnerability type, target characteristics, exploit success/failure).
    *   **Model Training**: Training a machine learning model (e.g., a reinforcement learning model or a predictive model) on this data to identify patterns, predict effective attack paths, or prioritize vulnerabilities.
    *   **Model Deployment**: Deploying the trained model to a Vertex AI Endpoint for online prediction.
    *   **Agent Improvement**: The agent would query this model to inform its next actions, such as choosing specific scanning techniques, prioritizing targets, or generating new exploit variations.
*   **Feedback Loop**: The success or failure of subsequent exploitation attempts (if implemented) would feed back into the learning model, creating a continuous improvement loop.
*   **Ethical Considerations**: Ensure the learning process always adheres to the defined scope and ethical guidelines. Implement strict controls to prevent the agent from acting outside its authorized boundaries.

This concludes the deployment and architectural overview. For further development, refer to individual agent directories and the `docs/` folder.
