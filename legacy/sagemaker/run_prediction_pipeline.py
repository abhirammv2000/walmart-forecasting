# run_prediction_pipeline.py (Final Corrected Version)

import sagemaker
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.model import Model
from sagemaker.transformer import Transformer
import boto3

# --- 1. CONFIGURE YOUR PIPELINE ---

# (REQUIRED) FILL THESE IN WITH YOUR VALUES
ROLE_ARN = 'arn:aws:iam::314146333022:role/SageMakerExecutionRole' 
S3_BUCKET_NAME = 'm5-walmart-forecast-data'
ECR_IMAGE_URI = '314146333022.dkr.ecr.us-east-2.amazonaws.com/m5-forecasting-image:latest'
# (CRITICAL) Paste the S3 URI to the model artifact from your successful training job
S3_MODEL_PATH = 's3://m5-walmart-forecast-data/models/m5-lgbm-direct-training-2025-07-17-22-24-19-329/output/model.tar.gz' # <-- PASTE YOUR S3 MODEL URI HERE

AWS_REGION = 'us-east-2'

# Give your jobs descriptive names
processing_job_name = 'm5-batch-feature-engineering'
transform_job_name = 'm5-batch-prediction'

# Define S3 paths for the pipeline steps
s3_raw_data_path = f's3://{S3_BUCKET_NAME}/data/raw/'
s3_feature_output_path = f's3://{S3_BUCKET_NAME}/features'
s3_prediction_output_path = f's3://{S3_BUCKET_NAME}/predictions'

# Define instance types
processing_instance_type = 'ml.m5.2xlarge'
transform_instance_type = 'ml.m5.2xlarge'

# --- 2. SETUP THE SAGEMAKER SESSION ---

boto_session = boto3.Session(region_name=AWS_REGION)
sagemaker_session = sagemaker.Session(boto_session=boto_session)

# --- 3. STEP 1: FEATURE ENGINEERING ---

print("--- Starting Step 1: Feature Engineering ---")

script_processor = ScriptProcessor(
    command=['python'],
    image_uri=ECR_IMAGE_URI,
    role=ROLE_ARN,
    instance_count=1,
    instance_type=processing_instance_type,
    base_job_name=processing_job_name,
    sagemaker_session=sagemaker_session
)

script_processor.run(
    code='scripts/feature_engineering.py', # The script to run
    inputs=[ProcessingInput(source=s3_raw_data_path, destination='/opt/ml/processing/input/data')],
    outputs=[ProcessingOutput(source='/opt/ml/processing/output/features', destination=s3_feature_output_path, output_name="features_for_prediction")]
)

print(f"Feature engineering complete. Features saved to {s3_feature_output_path}")

# --- 4. STEP 2: BATCH PREDICTION ---

print("\n--- Starting Step 2: Batch Prediction ---")

model = Model(
    image_uri=ECR_IMAGE_URI,
    model_data=S3_MODEL_PATH,
    role=ROLE_ARN,
    sagemaker_session=sagemaker_session,
    entry_point="predict.py"  # <-- This tells SageMaker to run 'python predict.py'
)

transformer = model.transformer(
    instance_count=1,
    instance_type=transform_instance_type,
    output_path=s3_prediction_output_path,
    # strategy and accept are not needed for Parquet and our custom container
)

transformer.transform(
    data=f"{s3_feature_output_path}/features_for_prediction",
    content_type='application/x-parquet',
    split_type='Line'
)

transformer.wait()

# --- THIS IS THE FIX ---
print(f"Batch prediction complete. Forecasts saved to {s3_prediction_output_path}")
print("\n--- PIPELINE FINISHED SUCCESSFULLY! ---")