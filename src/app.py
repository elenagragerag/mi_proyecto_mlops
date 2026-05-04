import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Credit Card Default Predictor")

with open("models/best_model.pkl", "rb") as f:
    model = pickle.load(f)

class ClientData(BaseModel):
    X1: float
    X2: float
    X3: float
    X4: float
    X5: float
    X6: float
    X7: float
    X8: float
    X9: float
    X10: float
    X11: float
    X12: float
    X13: float
    X14: float
    X15: float
    X16: float
    X17: float
    X18: float
    X19: float
    X20: float
    X21: float
    X22: float
    X23: float

@app.get("/")
def root():
    return {"message": "API de predicción de impago de tarjeta de crédito"}

@app.post("/predict")
def predict(data: ClientData):
    features = [[
        data.X1, data.X2, data.X3, data.X4, data.X5,
        data.X6, data.X7, data.X8, data.X9, data.X10,
        data.X11, data.X12, data.X13, data.X14, data.X15,
        data.X16, data.X17, data.X18, data.X19, data.X20,
        data.X21, data.X22, data.X23
    ]]
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]
    return {
        "prediction": int(prediction),
        "probability_of_default": round(float(probability), 4),
        "message": "Impago probable" if prediction == 1 else "Pago probable"
    }
