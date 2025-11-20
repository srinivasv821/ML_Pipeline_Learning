import os
import pandas as pd
import pickle
from sklearn.metrics import classification_report, accuracy_score
import mlflow

#set tracking uri for mlflow
mlflow.set_tracking_uri("http://ec2-43-204-214-202.ap-south-1.compute.amazonaws.com:5000")

# Get the project root directory
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# Define file paths safely
test_path = os.path.join(BASE_DIR, "data", "processed", "test.csv")
model_path = os.path.join(BASE_DIR, "models", "model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "models", "vectorizer.pkl")
report_dir = os.path.join(BASE_DIR, "reports")

# Ensure reports directory exists
os.makedirs(report_dir, exist_ok=True)

# Load test data and model artifacts
test = pd.read_csv(test_path)
model = pickle.load(open(model_path, "rb"))
vectorizer = pickle.load(open(vectorizer_path, "rb"))

# Transform data and evaluate
X_test = vectorizer.transform(test["message"])
y_test = test["label"]

pred = model.predict(X_test)

# Compute metrics
report = classification_report(y_test, pred, output_dict=True)
accuracy = accuracy_score(y_test, pred)

# Save metrics locally
report_path = os.path.join(report_dir, "metrics.txt")
with open(report_path, "w") as f:
    f.write(str(report))

# Log to MLflow
mlflow.set_experiment("sms_spam")
with mlflow.start_run():
    mlflow.log_metric("Accuracy", float(f"{accuracy:.4f}"))

print(f"✅ Evaluation done. Accuracy: {accuracy:.4f}")
print(f"📊 Metrics saved to: {report_path}")
