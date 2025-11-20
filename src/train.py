import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import mlflow
import mlflow.sklearn
import pickle

#set tracking uri for mlflow
mlflow.set_tracking_uri("http://ec2-43-204-214-202.ap-south-1.compute.amazonaws.com:5000")

# Get the project root directory
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# Build paths safely
train_path = os.path.join(BASE_DIR, "data", "processed", "train.csv")
model_dir = os.path.join(BASE_DIR, "models")

# Create model directory if missing
os.makedirs(model_dir, exist_ok=True)

# Load training data
train = pd.read_csv(train_path)
X_train = train["message"]
y_train = train["label"]

# Feature extraction
tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)

# Model training
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# MLflow experiment tracking
mlflow.set_experiment("sms_spam")
with mlflow.start_run(run_name="Test run - 1"):
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_metric("train_samples", len(train))
    mlflow.sklearn.log_model(model, "model")
    mlflow.sklearn.log_model(tfidf, "vectorizer")

# Save artifacts locally
model_path = os.path.join(model_dir, "model.pkl")
vectorizer_path = os.path.join(model_dir, "vectorizer.pkl")

pickle.dump(model, open(model_path, "wb"))
pickle.dump(tfidf, open(vectorizer_path, "wb"))

print(f"✅ Model trained and saved at: {model_path}")
