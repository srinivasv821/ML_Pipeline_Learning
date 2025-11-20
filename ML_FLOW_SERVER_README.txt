use this cmd to launch the mlflow server on EC2

mlflow server \
  --backend-store-uri postgresql://mlflowuser:recyclebin123@mlflow-db.cp4aakc64yua.ap-south-1.rds.amazonaws.com:5432/mlflow \
  --default-artifact-root s3://mlflow-dvc-dev-store/mlflow-artifacts \
  --host 0.0.0.0 \
  --port 5000 \
  --allowed-hosts '*' \
  --cors-allowed-origins '*'

