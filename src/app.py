from fastapi import FastAPI
import mlflow
import mlflow.sklearn
import pickle
from pydantic import BaseModel
import os

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
model_path = os.path.join(BASE_DIR, "models", "model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "models", "vectorizer.pkl")

model = pickle.load(open(model_path, "rb"))
vectorizer = pickle.load(open(vectorizer_path, "rb"))

mlflow.set_tracking_uri("http://ec2-43-204-214-202.ap-south-1.compute.amazonaws.com:5000")

class Message(BaseModel):
    text:str

@app.post("/predict")
def predict(msg: Message):
    # Convert to vector
    x = vectorizer.transform([msg.text])
    # Predict
    y = model.predict(x)[0]

    return {"prediction": str(y)}