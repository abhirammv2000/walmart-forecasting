# run_training_job.py (Final Corrected Version)

import sagemaker
from sagemaker.estimator import Estimator
import boto3

# --- 1. CONFIGURE YOUR JOB ---

# (REQUIRED) FILL THESE IN WITH YOUR VALUES
ROLE_ARN = 'arn:aws:iam::314146333022:role/SageMakerExecutionRole' 
S3_BUCKET_NAME = 'm5-walmart-forecast-data'
ECR_IMAGE_URI = '314146333022.dkr.ecr.us-east-2.amazonaws.com/m5-forecasting-image:latest'

# --- KEY FIX: Specify the AWS region to match your ECR repository ---
AWS_REGION = 'us-east-2'

# Give your training job a descriptive name
job_name = 'm5-lgbm-direct-training'

# Define the S3 paths for input data and output models
s3_input_path = f's3://{S3_BUCKET_NAME}/data/raw/'
s3_output_path = f's3://{S3_BUCKET_NAME}/models/'

# Define the instance type and count.
instance_type = 'ml.m5.4xlarge'
instance_count = 1

# --- 2. CREATE A SAGEMAKER SESSION IN THE CORRECT REGION ---

# Create a boto3 session to specify the region
boto_session = boto3.Session(region_name=AWS_REGION)

# Create a sagemaker session from the boto3 session
sagemaker_session = sagemaker.Session(boto_session=boto_session)


# --- 3. CREATE THE SAGEMAKER ESTIMATOR ---

estimator = Estimator(
    image_uri=ECR_IMAGE_URI,
    role=ROLE_ARN,
    instance_count=instance_count,
    instance_type=instance_type,
    output_path=s3_output_path,
    base_job_name=job_name,
    sagemaker_session=sagemaker_session, # Pass the region-specific session here
    max_run=3600 * 18, # 6 hours
)

# --- 4. LAUNCH THE JOB ---

print(f"Starting SageMaker Training Job '{job_name}' in region '{AWS_REGION}'...")
print(f"  - Instance Type: {instance_type}")
print(f"  - Input Data: {s3_input_path}")
print(f"  - Model Output: {s3_output_path}")

estimator.fit({'data': s3_input_path})

print("Job launched successfully! You can monitor its progress in the AWS SageMaker Console.")