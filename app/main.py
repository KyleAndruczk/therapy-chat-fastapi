from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.model import MentalHealthClassifier


app = FastAPI(title="Mental Health Problem Classifier")

# Initialize the classifier
classifier = MentalHealthClassifier()

class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    predicted_topic: str
    likelihood: float

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if not request.text:
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    prediction = classifier.predict(request.text)
    return prediction

@app.get("/")
def read_root():
    return {"message": "Welcome to the Mental Health Problem Classifier API. Use the /predict endpoint to get predictions."}
