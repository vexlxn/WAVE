from fastapi import FastAPI, Form
import pickle
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

# Load model and scaler
with open("water_usage_model.pkl", "rb") as f:  # ✅ correct filename
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Initialize FastAPI
app = FastAPI()

# Allow CORS for frontend testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict/")
def predict(
    family_member: float = Form(...),
    weekly: float = Form(...),
    monthly: float = Form(...),
    by_cycle: float = Form(...),
    seasons: float = Form(...)  # 1=winter, 2=spring, etc.
):
    features = np.array([[family_member, weekly, monthly, by_cycle, seasons]])
    scaled = scaler.transform(features)
    prediction = model.predict(scaled)
    return {"predicted_bill": round(prediction[0], 2)}
