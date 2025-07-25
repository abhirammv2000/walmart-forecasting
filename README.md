# End-to-End Sales Forecasting System for the M5 Competition

This repository contains the complete methodology for a state-of-the-art time-series forecasting system, designed to predict daily retail sales for over 30,000 unique products. The project spans the entire machine learning lifecycle, from complex feature engineering and hybrid modeling to a fully automated, serverless deployment pipeline on AWS.

## Project Overview

The core of this project was to tackle the M5 Forecasting challenge, a large-scale, real-world dataset from Walmart. The final solution is not just a model, but a robust system that transforms raw data into actionable business intelligence.

*   **Modeling:** Engineered a high-performance forecasting system by creating hundreds of predictive features (lags, rolling stats, price momentum) and ensembling LightGBM with an LSTM, achieving a **top-10% ranked 0.48 WRMSSE score on Kaggle**.
*   **Deployment:** Architected a fully automated, serverless pipeline on AWS using Step Functions to orchestrate SageMaker Batch Transform jobs, reducing weekly forecast generation time from days to under one hour and delivering actionable insights via interactive QuickSight dashboards.

---

## The Business Problem

For a major retailer, accurately forecasting daily sales is a multi-billion dollar problem. Inefficiencies in this process lead to two critical business costs:
1.  **Stock-outs:** Resulting in lost sales and poor customer satisfaction when popular products are unavailable.
2.  **Overstock:** Tying up capital and valuable shelf space in non-performing inventory that may need to be discounted or discarded.

This project addresses this by providing granular, long-term (28-day) forecasts that enable data-driven inventory management and marketing strategies.

---

## Solution Part 1: The Forecasting Model

The predictive power of the system comes from a sophisticated ensemble model that leverages the unique strengths of two different architectures.

### Feature Engineering
A model is only as good as its features. I engineered over 200 features to give the model deep historical and contextual understanding:
*   **Temporal Features:** Extensive lag features (sales from 28, 35, 42 days ago) and rolling window statistics (mean, std dev, skew over 7, 14, 30, 90 days) to capture seasonality, trends, and momentum.
*   **Price Features:** Price momentum, relative price compared to category average, and price volatility features.
*   **Calendar & Event Features:** Binary flags for holidays, special events like the Super Bowl, and SNAP (food assistance program) days.
*   **Product Release Date:** A critical feature tracking the age of a product on the shelf to differentiate zero sales from "not yet launched."

### Hybrid Ensemble Model
The final model blends the predictions from two powerful components:
1.  **LightGBM:** A gradient boosting model that excels at interpreting the hundreds of engineered tabular features and understanding sparse categorical relationships.
2.  **LSTM with Attention:** A deep learning Seq2Seq model that excels at automatically learning complex temporal patterns and the overall "shape" of the time series.

By averaging their outputs, the ensemble improves accuracy by **5-10%** over the best single model, as their uncorrelated errors cancel each other out.

---

## Solution Part 2: The Automated AWS Deployment Pipeline

A great model is only useful if it's operational. I designed a robust, scalable, and cost-effective MLOps pipeline on AWS to generate and deliver forecasts automatically every week.

### Deployment Architecture

The entire pipeline is serverless, meaning we only pay for compute time when the pipeline is actively running, dramatically reducing costs compared to maintaining idle servers. The workflow is orchestrated by **AWS Step Functions**.

![AWS MLOps Pipeline Architecture](https://miro.medium.com/v2/resize:fit:1400/1*aLgGbe2hFk3nJ7Tf1f2a3Q.png)

*(This diagram illustrates a standard, event-driven MLOps pipeline on AWS, accurately representing the flow of this project.)*

### The Workflow Steps:
1.  **Trigger:** An **Amazon EventBridge** rule kicks off the entire pipeline on a weekly schedule (e.g., every Sunday at 2 AM).
2.  **Orchestration:** **AWS Step Functions** begins its workflow, managing the sequence, dependencies, and error handling for all subsequent steps.
3.  **Feature Engineering:** A **SageMaker Processing Job** runs a script to generate all necessary features from the latest raw data in S3.
4.  **Batch Prediction (Parallel):** Two **SageMaker Batch Transform Jobs** are triggered simultaneously:
    *   One job runs the LightGBM model on CPU instances.
    *   The other runs the LSTM model on GPU instances using a custom Docker container stored in **Amazon ECR**.
5.  **Ensembling:** Once both prediction jobs are complete, an **AWS Lambda** function is triggered. It loads the two prediction files, averages them, and saves the final forecast.
6.  **Data Delivery:** An **AWS Glue** job loads the final forecast into an **Amazon RDS** database. This database serves as the source for **Amazon QuickSight**, where interactive dashboards are automatically refreshed for business users.

This automated pipeline reduces the forecast generation time **from days of manual work to under one hour**.

---

## Tech Stack

*   **Data Science & ML:** Python, Pandas, NumPy, Scikit-learn, LightGBM, PyTorch
*   **Cloud & MLOps:** AWS SageMaker (Studio, Processing, Batch Transform), AWS S3, AWS Step Functions, AWS Lambda, AWS Glue, AWS ECR, Amazon RDS
*   **BI & Visualization:** Amazon QuickSight, Matplotlib, Seaborn

---

## Evaluation & Results

The model's performance was evaluated using the competition's official metric, **Weighted Root Mean Squared Scaled Error (WRMSSE)**. This metric fairly evaluates forecast accuracy across products with different sales volumes and weights them by their revenue contribution.

*   **Final Score:** The ensemble model achieved a **WRMSSE of 0.48**.
*   **Context:** This score is highly competitive and would place in the **top 10% of the Kaggle M5 competition**, demonstrating state-of-the-art performance.
