#!/bin/bash
set -e

echo "Starting MLflow server..."

# Install MLflow and dependencies if not present
pip install --no-cache-dir \
    mlflow==2.8.0 \
    boto3 \
    psycopg2-binary \
    prometheus-flask-exporter

# Wait for PostgreSQL to be ready
echo "Waiting for PostgreSQL..."
while ! nc -z postgres 5432; do
    sleep 1
done
echo "PostgreSQL is ready!"

# Wait for MinIO to be ready
echo "Waiting for MinIO..."
while ! nc -z minio 9000; do
    sleep 1
done
echo "MinIO is ready!"

# Configure MLflow
export MLFLOW_S3_ENDPOINT_URL=http://minio:9000
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin123

# Create MLflow bucket in MinIO if it doesn't exist
python -c "
import boto3
from botocore.client import Config

s3 = boto3.client(
    's3',
    endpoint_url='http://minio:9000',
    aws_access_key_id='minioadmin',
    aws_secret_access_key='minioadmin123',
    config=Config(signature_version='s3v4')
)

try:
    s3.create_bucket(Bucket='mlflow')
    print('Created mlflow bucket')
except Exception as e:
    print(f'Bucket might already exist: {e}')
"

# Start MLflow server
mlflow server \
    --backend-store-uri postgresql://mlflow:mlflow123@postgres:5432/mlflow \
    --default-artifact-root s3://mlflow/artifacts \
    --host 0.0.0.0 \
    --port 5000 \
    --serve-artifacts
