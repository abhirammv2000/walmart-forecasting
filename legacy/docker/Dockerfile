# Dockerfile (Final Version for Deployment)

FROM python:3.9-slim
WORKDIR /opt/ml/code

# Install system dependencies required by LightGBM
RUN apt-get update && apt-get install -y libgomp1 && \
    rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copy requirements from the 'docker' subfolder
COPY docker/requirements.txt .

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy all scripts from the 'scripts' subfolder
COPY scripts/ .

# NO ENTRYPOINT - This makes the container flexible