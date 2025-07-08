# M5 Forecasting: A Production-Grade MLOps Pipeline on Google Cloud

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)
![Python Version](https://img.shields.io/badge/python-3.9-blueviolet)

## Overview

This repository contains the complete code and documentation for an end-to-end machine learning project that tackles the [M5 Forecasting challenge](https://www.kaggle.com/c/m5-forecasting-accuracy). The primary goal was to build a robust, automated system that accurately forecasts daily sales for thousands of Walmart products across multiple states.

The project goes beyond a simple model in a notebook. It implements a full MLOps pipeline on **Google Cloud Platform (GCP)**, demonstrating how to handle large-scale data, train models in the cloud, deploy serverless prediction services, and automate the entire forecasting process.

### Core Features

*   **End-to-End Solution:** Covers the entire project lifecycle, from data pre-processing to a live, automated forecasting pipeline.
*   **Scalable Model Training:** Uses **Vertex AI** to train a LightGBM model on a massive dataset (58+ million rows) that would be impossible to handle on a local machine.
*   **Serverless Prediction API:** Deploys the forecasting logic as a containerized application on **Cloud Run**, Google's serverless platform.
*   **Fully Automated Pipeline:** Uses **Cloud Scheduler**, **Pub/Sub**, and **Eventarc** to automatically trigger a new forecast every week without any manual intervention.
*   **Data Warehousing:** Stores the final forecast results in **BigQuery**, providing a structured, queryable source of truth.
*   **Interactive Dashboard:** Demonstrates how to connect **Looker Studio** to BigQuery to create a user-friendly dashboard for exploring the forecast data.

## Final System Architecture

The final deployment is an event-driven, serverless architecture that is both scalable and cost-effective.

```
┌───────────────────┐ ┌──────────┐ ┌─────────────────┐ ┌────────────────────────────┐
│ Cloud Scheduler   ├─────►│ Pub/Sub ├─────►│ Eventarc Trigger ├─────►│ Cloud Run Service         │
│ (Weekly Cron Job) │     │ Topic   │     │ (Listens for msg) │     │ (m5-batch-forecast-trigger) │
└───────────────────┘ └──────────┘ └─────────────────┘ └──────────┬──────────┬────────┘
                                                                  │          │
                                                                  ▼          ▼
                                                     ┌──────────┴────────┐ ┌────────────────┐
                                                     │ GCS Bucket        │ │ BigQuery       │
                                                     │ (Model & Raw Data)│◄─┤ (Forecast Sink)│
                                                     └───────────────────┘ └────────────────┘

```

**Workflow:**
1.  **Cloud Scheduler** fires a cron job every week.
2.  The job sends a message to a **Pub/Sub** topic.
3.  An **Eventarc** trigger detects the message and invokes the **Cloud Run** service.
4.  The Cloud Run service downloads the trained model and raw data from **Google Cloud Storage (GCS)**, runs the 28-day recursive forecast, and saves the results to **BigQuery**.
5.  A **Looker Studio** dashboard connects to BigQuery to visualize the forecast.

## Project Structure

The repository is organized into two main components: `training` and `prediction_server`.

```
walmart-forecasting/
│
├── training/
│   ├── Dockerfile         # Defines the environment for the Vertex AI training job.
│   ├── train.py           # The Python script that trains the LightGBM model.
│   └── config.yaml        # Configuration file for the Vertex AI Custom Job.
│
├── prediction_server/
│   ├── Dockerfile         # Defines the environment for the Cloud Run prediction service.
│   ├── main.py            # The Python script with the main prediction logic.
│   └── requirements.txt   # Python dependencies for the prediction service.
│
├── data/                  # (Local) Holds the raw M5 competition CSV files.
│
└── README.md              # This file.
```

## How to Deploy: Step-by-Step Guide

Follow these steps to deploy the entire pipeline from scratch.

### 1. Prerequisites
-   A Google Cloud Platform (GCP) project with billing enabled.
-   The `gcloud` command-line tool installed and authenticated (`gcloud auth login`).
-   Docker installed and running on your local machine.
-   The raw M5 data files placed in a local `data/` directory.

### 2. GCP Setup
1.  **Enable APIs:** Enable the following APIs in your GCP project: Vertex AI, Cloud Storage, BigQuery, Cloud Build, Cloud Run, Eventarc, Pub/Sub, and Cloud Scheduler.
2.  **Create a GCS Bucket:** Choose a unique name for your bucket.
    ```bash
    gsutil mb -p [YOUR_PROJECT_ID] -l US-CENTRAL1 gs://[YOUR_BUCKET_NAME]
    ```
3.  **Upload Raw Data:**
    ```bash
    gsutil -m cp -r ./data/* gs://[YOUR_BUCKET_NAME]/data/
    ```

### 3. Phase I: Train the Model on Vertex AI
1.  **Navigate to the Training Directory:**
    ```bash
    cd training
    ```
2.  **Edit `config.yaml`:** Replace the `[YOUR_PROJECT_ID]` and `[YOUR_BUCKET_NAME]` placeholders with your actual values.
3.  **Build the Training Container:**
    ```bash
    gcloud builds submit --tag gcr.io/[YOUR_PROJECT_ID]/m5-stable-trainer:latest .
    ```
4.  **Launch the Vertex AI Training Job:**
    ```bash
    gcloud ai custom-jobs create \
      --project=[YOUR_PROJECT_ID] \
      --region=us-central1 \
      --display-name="m5-stable-model-training" \
      --config=config.yaml
    ```
5.  **Wait for Success:** Monitor the job in the Vertex AI console. It will take over an hour to complete. Once it succeeds, your trained model (`m5_stable_model.txt`) will be in your GCS bucket under `model_artifacts/`.

### 4. Phase II: Deploy the Prediction Service
1.  **Navigate to the Prediction Directory:**
    ```bash
    cd ../prediction_server
    ```
2.  **Build the Prediction Container:**
    ```bash
    gcloud builds submit --tag gcr.io/[YOUR_PROJECT_ID]/m5-prediction-server:latest .
    ```
3.  **Deploy to Cloud Run:** This deploys your service privately.
    ```bash
    gcloud run deploy m5-batch-forecast-trigger \
      --project=[YOUR_PROJECT_ID] \
      --region=us-central1 \
      --image=gcr.io/[YOUR_PROJECT_ID]/m5-prediction-server:latest \
      --set-env-vars=GCP_PROJECT=[YOUR_PROJECT_ID],GCS_BUCKET=[YOUR_BUCKET_NAME],BQ_DATASET=m5_forecasts,BQ_TABLE=daily_forecasts \
      --timeout=540s \
      --memory=4Gi \
      --cpu=2 \
      --no-allow-unauthenticated
    ```

### 5. Phase III: Configure Automation
1.  **Create BigQuery Resources:**
    ```bash
    bq mk --dataset [YOUR_PROJECT_ID]:m5_forecasts
    bq mk --table [YOUR_PROJECT_ID]:m5_forecasts.daily_forecasts id:STRING,d:INTEGER,forecast_sales:FLOAT,forecast_timestamp:TIMESTAMP
    ```
2.  **Create Pub/Sub Topic:**
    ```bash
    gcloud pubsub topics create m5-forecast-run-topic --project=[YOUR_PROJECT_ID]
    ```
3.  **Create Service Account and Grant Permissions:**
    ```bash
    gcloud iam service-accounts create m5-eventarc-invoker --project=[YOUR_PROJECT_ID]
    gcloud run services add-iam-policy-binding m5-batch-forecast-trigger \
      --project=[YOUR_PROJECT_ID] \
      --region=us-central1 \
      --member="serviceAccount:m5-eventarc-invoker@[YOUR_PROJECT_ID].iam.gserviceaccount.com" \
      --role="roles/run.invoker"
    ```
4.  **Create the Eventarc Trigger:**
    ```bash
    gcloud eventarc triggers create m5-forecast-trigger \
      --project=[YOUR_PROJECT_ID] \
      --location=us-central1 \
      --destination-run-service=m5-batch-forecast-trigger \
      --event-filters="type=google.cloud.pubsub.topic.v1.messagePublished" \
      --transport-topic=m5-forecast-run-topic \
      --service-account="m5-eventarc-invoker@[YOUR_PROJECT_ID].iam.gserviceaccount.com"
    ```
5.  **Create the Cloud Scheduler Job:**
    ```bash
    gcloud scheduler jobs create pubsub run-weekly-m5-forecast \
      --project=[YOUR_PROJECT_ID] \
      --schedule="0 1 * * 1" \
      --topic=m5-forecast-run-topic \
      --message-body="Run forecast" \
      --time-zone="America/New_York"
    ```

### 6. Phase IV: Test and Visualize
1.  **Manually trigger the pipeline** by publishing a message to the `m5-forecast-run-topic` in the GCP console.
2.  **Monitor the logs** in Cloud Run to watch the execution.
3.  **Verify the data** in the BigQuery table.
4.  **Connect Looker Studio** to your BigQuery table and build your interactive dashboard.

## Key Learnings from this Project

*   **Cloud Architecture Matters:** The most significant challenges were not in the model itself, but in designing a cloud architecture that could handle the scale and constraints of the environment (e.g., solving memory errors, startup timeouts, and IAM permissions).
*   **Stability over Complexity:** The simplest model (a single LightGBM) proved to be the most robust and highest-scoring in production, outperforming more complex ensemble methods that were prone to overfitting and bugs.
*   **Debugging is Iterative:** Deploying to the cloud is a process of inches. Each failed run provides a crucial log message that points to the next configuration fix. Persistence and a systematic approach are key.
*   **Infrastructure as Code is Superior:** Using `config.yaml` and `gcloud` commands is more reliable, repeatable, and less error-prone than clicking through a complex user interface.

