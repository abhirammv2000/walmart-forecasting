# End-to-End Hierarchical Demand Forecasting System

This repository contains the complete methodology for a state-of-the-art demand forecasting system, designed to predict daily unit sales for over **30,490 unique products** across Walmart's network. The project transforms raw retail data into actionable inventory insights using a hierarchical LightGBM model and a fully automated AWS pipeline.

## Project Overview

The core of this project addresses the **M5 Forecasting** challenge: predicting sales for intermittent, high-volatility retail items. The solution prioritizes scalability and interpretability, moving beyond simple accuracy to solve the "zero-sales" problem inherent in supply chain data.

* **Modeling:** Engineered a hierarchical demand forecasting engine using **LightGBM with Tweedie loss**, specifically designed to handle intermittent (zero-inflated) demand. The model achieved a **Validation WRMSSE of 0.58**, a performance benchmark comparable to the **Top 5%** of the global leaderboard.
* **Deployment:** Architected a serverless batch inference pipeline on AWS using **Step Functions** to orchestrate **SageMaker Batch Transform**, automating weekly forecasts and integrating directly with **Amazon QuickSight** for inventory planning.

---

## The Business Problem

For major retailers, "Sales Forecasting" is often insufficient because it fails to account for lost demand during stockouts. This project focuses on **Demand Forecasting** to mitigate two multi-billion dollar risks:
1.  **Stock-outs:** Preventing lost revenue by predicting demand spikes before they empty the shelves.
2.  **Overstocking:** Reducing holding costs for slow-moving items with intermittent sales patterns.

The system provides granular, 28-day forecasts at the SKU level while maintaining consistency across stores and states.

---

## Solution Part 1: The Forecasting Model

Unlike traditional regression models that struggle with sparse data, this solution uses a gradient-boosting approach optimized for retail characteristics.

### 1. Handling Intermittent Demand (The "Tweedie" Advantage)
Retail data is "zero-inflated"—many items don't sell every single day. Standard RMSE loss functions treat these zeros as noise, leading to under-forecasting.

* **Objective Function:** I utilized **Tweedie Loss** (variance power $1 < p < 2$), which models a compound Poisson-Gamma distribution. This allows the model to simultaneously predict *if* an item will sell (probability of zero) and *how much* it will sell (magnitude).
* **Result:** This significantly improved accuracy on slow-moving inventory compared to standard Poisson or RMSE objectives.

### 2. Feature Engineering
I engineered over 50 robust features to capture temporal dynamics and pricing psychology:
* **Lag Features:** Sales from shifted windows (e.g., `lag_7`, `lag_28`) to capture weekly seasonality.
* **Rolling Statistics:** Moving averages and standard deviations over 7, 30, and 90-day windows to detect trend stability.
* **Price Momentum:** Relative price changes (Current Price vs. Historical Average) to measure price elasticity.
* **Calendar Events:** Binary flags for SNAP (food stamps) release dates, holidays, and sporting events.

---

## Solution Part 2: The Automated AWS Deployment Pipeline

The model is deployed via a **serverless, event-driven architecture** on AWS. This design decouples compute from storage, ensuring the system costs near-zero when not actively generating forecasts.

### Deployment Architecture
The pipeline is orchestrated by **AWS Step Functions**, which manages the workflow state, retries, and error handling.

### The Workflow Steps:
1.  **Trigger:** An **Amazon EventBridge** rule triggers the pipeline on a weekly schedule (e.g., Sunday 2 AM) or upon new data arrival in S3.
2.  **Orchestration:** **AWS Step Functions** initializes the state machine.
3.  **Batch Inference:** The workflow triggers a **SageMaker Batch Transform** job.
    * It spins up transient compute instances (e.g., `ml.m5.xlarge`).
    * It processes the 30K+ SKU feature vectors in parallel against the trained LightGBM model.
    * It saves the raw predictions back to a private S3 bucket.
4.  **Post-Processing:** An **AWS Lambda** function triggers to validate the output file and format the results (adding date timestamps and SKU identifiers).
5.  **Visualization:** The final dataset in S3 is ingested by **Amazon QuickSight** (via SPICE). Dashboards automatically refresh, presenting "Stockout Risk" and "Predicted Demand" views to stakeholders.

---

## Tech Stack

* **Modeling:** Python, LightGBM (Tweedie Objective), Pandas, NumPy, Scikit-Learn
* **Cloud Infrastructure:** AWS Step Functions, AWS SageMaker (Batch Transform), AWS Lambda, Amazon S3, Amazon EventBridge
* **Analytics & BI:** Amazon QuickSight

---

## Evaluation & Results

The system was evaluated using **WRMSSE** (Weighted Root Mean Squared Scaled Error), a metric that penalizes errors on high-value items more heavily than low-value ones.

* **Validation Performance:** The model achieved a **WRMSSE of 0.58**.
* **Impact:** This accuracy outperforms the seasonal naive baseline by **>40%** and aligns with the top 5% of solutions on the global leaderboard, demonstrating the effectiveness of the Tweedie loss formulation for large-scale retail data.
