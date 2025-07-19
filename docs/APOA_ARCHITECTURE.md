# Aetherveil - Autonomous Pipeline Observer Agent (APOA) Architecture

## 1. Overview

The Autonomous Pipeline Observer Agent (APOA) is a core component of the Aetherveil Autonomous DevOps platform. It is a GCP-native service designed to provide real-time monitoring, anomaly detection, and actionable insights for CI/CD pipelines.

The agent operates by ingesting data from various sources, processing it to build performance baselines, and using machine learning models to detect deviations that could indicate failures, performance regressions, or security anomalies.

## 2. Core Principles

- **Event-Driven:** The system is asynchronous and driven by events from CI/CD workflows and GCP services. It avoids polling.
- **Stateless & Scalable:** The agent is designed as a stateless Cloud Run service, allowing it to scale horizontally to handle variable loads.
- **Pluggable Intelligence:** The analysis engine is decoupled from the core data processing pipeline, allowing for different ML models (from Vertex AI AutoML to custom models) to be used.
- **Secure by Design:** Leverages GCP's Workload Identity Federation for keyless, secure authentication between GitHub Actions and GCP services.

## 3. High-Level Architecture Diagram

```mermaid
graph TD
    subgraph "Data Sources"
        A[GitHub Actions Workflows]
        B[GCP Cloud Logging]
        C[GCP Cloud Run/GKE Metrics]
    end

    subgraph "GCP Ingestion Layer"
        P1[Pub/Sub Topic: github-events]
        P2[Pub/Sub Topic: pipeline-logs]
        P3[Pub/Sub Topic: pipeline-metrics]
    end

    subgraph "APOA Cloud Run Service"
        direction LR
        SUB[Subscriber Module] --> PROC[Data Processor]
        PROC --> |Normalized Data| SI[Storage Interface]
        PROC --> |Real-time Data| AE[Analysis Engine]
        AE --> |Historical Query| SI
        AE --> |Prediction Request| VAI[Vertex AI Endpoint]
        VAI --> |Anomaly Score| AE
        AE --> |Insight| IG[Insight Generator]
    end

    subgraph "GCP Data & AI"
        direction TB
        BQ[BigQuery Dataset: pipeline_analytics]
        VAI(Vertex AI Model Endpoint)
    end

    subgraph "Action & Notification Layer"
        P4[Pub/Sub Topic: apoa-findings]
        P4 --> N1[Notification Service (e.g., Slack)]
        P4 --> N2[GitHub Issue Creator]
        P4 --> N3[Dashboard Service]
    end

    %% Data Flow
    A -- Webhook --> P1
    B -- Log Sink --> P2
    C -- Metrics Sink --> P3

    P1 & P2 & P3 --> SUB

    SI -- Stores/Queries --> BQ
    IG -- Publishes --> P4

    classDef service fill:#e6f2ff,stroke:#b3d9ff,stroke-width:2px;
    class APOA,VAI,BQ service;
```

## 4. Component Breakdown

### 4.1. Data Sources
- **GitHub Actions:** Workflows (`reusable-ci.yml`, etc.) are the primary source of pipeline events. We will use webhooks to capture `workflow_job` and `workflow_run` events, which provide context about job status, timing, and conclusion.
- **GCP Cloud Logging:** A centralized sink captures logs from all pipeline steps, including build, test, and deployment processes. This provides the raw data for in-depth failure analysis.
- **GCP Metrics:** Cloud Run and GKE metrics provide data on resource utilization (CPU, memory) of the pipeline jobs themselves, which is crucial for detecting performance bottlenecks and resource misuse.

### 4.2. Ingestion Layer (GCP Pub/Sub)
- **`github-events` (Pub/Sub Topic):** A dedicated topic to receive webhook payloads from GitHub. An intermediary service (like a simple Cloud Function) will be needed to receive the public webhook and publish it securely to this topic.
- **`pipeline-logs` (Pub/Sub Topic):** Receives log entries from a configured GCP Logging sink. The sink will be filtered to only include logs relevant to CI/CD runs.
- **`pipeline-metrics` (Pub/Sub Topic):** Receives metrics from a configured GCP Monitoring sink.

### 4.3. APOA Service (Cloud Run)
This is the core of the system, containerized and deployed as a stateless Cloud Run service.

- **Subscriber Module:** Contains push subscribers for the three ingestion topics. It receives raw data and passes it to the Data Processor.
- **Data Processor:**
    - Parses and normalizes incoming JSON payloads (from Pub/Sub) into a standardized schema.
    - Enriches data, e.g., linking a log entry to a specific GitHub workflow run ID.
    - Forwards the normalized data to the Storage Interface and the Analysis Engine.
- **Storage Interface:**
    - Abstracts the connection to the data backend.
    - Its primary responsibility is to stream processed data into a BigQuery table for long-term storage and historical analysis.
- **Analysis Engine:**
    - Orchestrates the anomaly detection process.
    - On receiving new data (e.g., a completed job), it queries BigQuery via the Storage Interface to fetch historical performance baselines for that specific job.
    - It sends the combined real-time and historical data to a Vertex AI model endpoint for prediction.
- **Insight Generator:**
    - Receives the analysis result (e.g., anomaly score, feature attribution) from the Analysis Engine.
    - Formats this into a human-readable insight and a machine-readable action.
    - Publishes the finding to the `apoa-findings` Pub/Sub topic.

### 4.4. Data & AI (GCP)
- **BigQuery (`pipeline_analytics`):** A dataset used to store all historical pipeline execution data. This serves as the "memory" of the system and the source of truth for training ML models and establishing performance baselines.
- **Vertex AI:** Hosts the machine learning models. Initially, this can be an AutoML model trained on the BigQuery data to detect anomalies. The decoupled design allows for swapping this with more sophisticated custom models in the future.

### 4.5. Action & Notification Layer
- **`apoa-findings` (Pub/Sub Topic):** A broadcast topic for all insights generated by the APOA. This decouples the agent from the specific actions taken in response to its findings.
- **Downstream Subscribers:** Various tools can subscribe to this topic to take action:
    - A Cloud Function that formats the finding and sends a **Slack/MSTeams message**.
    - A service that creates a **GitHub Issue** with detailed context for a developer to investigate.
    - A service that updates a **Grafana/Looker Studio dashboard** with real-time pipeline health.

## 5. Security & Authentication
- **GitHub → GCP:** Workload Identity Federation is used. The GitHub Actions workflow will be configured to assume a specific GCP Service Account, granting it temporary, keyless permission to publish to the `github-events` Pub/Sub topic.
- **APOA Service Account:** The Cloud Run service will run under a dedicated IAM Service Account with the minimum required permissions:
    - Pub/Sub Subscriber/Publisher
    - BigQuery Data Editor
    - Vertex AI User
    - Cloud Trace User (for observability)
