# TensorFlow Extended (TFX) Pipeline for Loan Approval Prediction

## Overview
This project implements a machine learning pipeline using TensorFlow Extended (TFX) for loan approval prediction. It uses a synthetic dataset, processes the data, trains a model, and serves the model for predictions through a cloud-based deployment.

---

## Project Structure

### **Folder and File Descriptions**

#### 1. **config/**
- Contains the configuration file for Prometheus (`prometheus.config`).

#### 2. **data/**
- Stores the dataset in CSV format.
- Dataset used: [Loan Approval Classification Dataset](https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data).
  - **Description**: Synthetic dataset based on credit risk data, consisting of 45,000 records and 14 variables.

#### 3. **images/**
- Contains image files (not relevant for the pipeline).

#### 4. **modules/**
- **loan_transform.py**: Module for the Transform component of TFX.
- **loan_trainer.py**: Module for hyperparameter tuning and model training within TFX.

#### 5. **monitoring/**
- Dockerfile and configuration for monitoring services.
  - **prometheus.yml**: Configuration file for Prometheus.
  - **Grafana/**: Contains the following files:
    - `tensorflow-serving-dashboard.json`: Dashboard configuration for TensorFlow Serving in Grafana.
    - `prometheus-datasource.yml`: Data source configuration for Prometheus in Grafana.

#### 6. **output/**
- Stores the output directory of the TFX pipeline, including:
  - `serving_model`: Saved models ready for deployment.
  - `tuning_results`: Results from the hyperparameter tuning process.

#### 7. **Dockerfile**
- Dockerfile for deploying the trained model to the cloud using [Railway](https://railway.app/).

#### 8. **loan_pipeline.ipynb**
- Jupyter Notebook containing the TensorFlow Extended pipeline.
  - Implements components for data ingestion, transformation, model training, evaluation, and deployment.

#### 9. **test.ipynb**
- Jupyter Notebook for testing the API of the deployed model.
  - Validates the integration and prediction service deployed on Railway.

---

## Pipeline Description
The pipeline leverages TensorFlow Extended to create a robust, production-ready machine learning system. The key components include:

1. **Data Ingestion**
   - Reads and validates the input CSV data.

2. **Data Transformation**
   - Applies preprocessing using the `loan_transform.py` module.

3. **Trainer Component**
   - Trains the model with hyperparameter tuning using the `loan_trainer.py` module.

4. **Evaluator Component**
   - Evaluates the model’s performance using validation data.

5. **Pusher Component**
   - Deploys the trained model for serving.

6. **Model Deployment**
   - Deployed using a Docker container and served through the Railway cloud platform.

---

## Monitoring Setup

1. **Prometheus**
   - Collects metrics from the deployed model for monitoring.

2. **Grafana**
   - Visualizes metrics using dashboards like `tensorflow-serving-dashboard.json`.

---

## How to Run the Project

### 1. **Setup**
- Install dependencies from `requirements.txt`.
- Ensure Docker is installed and configured.

### 2. **Run the TFX Pipeline**
- Execute `loan_pipeline.ipynb` to run the TensorFlow Extended pipeline.

### 3. **Model Deployment**
- Build and run the Docker container using the outer `Dockerfile`.
- Deploy the container to Railway or any other cloud platform.

### 4. **Monitoring**
- Configure Prometheus using `monitoring/prometheus.yml`.
- Start Grafana and load the dashboard from `monitoring/Grafana/tensorflow-serving-dashboard.json`.

### 5. **API Testing**
- Use `test.ipynb` to test the deployed model’s API.

---

## Future Work
- Enhance the monitoring setup by integrating advanced alerting mechanisms.
- Experiment with additional preprocessing and feature engineering techniques to improve model performance.
- Explore alternative deployment platforms for scalability.

---

## Acknowledgments
- Dataset Source: [Loan Approval Classification Dataset](https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data)
- Tools: TensorFlow Extended, Prometheus, Grafana, Docker, Railway

---

Feel free to reach out for further questions or improvements regarding this project!