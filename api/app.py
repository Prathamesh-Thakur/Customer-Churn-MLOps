from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from pathlib import Path
from datetime import datetime, UTC
import csv
import joblib

from src.predict import make_prediction

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_FILE = BASE_DIR / "data" / "live_inference_logs.csv"
MODEL_PATH = BASE_DIR / "models" / "training_model.joblib"
feature_file_path = BASE_DIR / "models" / "imp_features.txt"

app = FastAPI(
    title="Telco Churn Prediction API", 
    description="An API that predicts the probability of a customer churning.",
    version="1.0"
)

# Define data schema
class CustomerData(BaseModel):
    tenure: int
    Contract: str
    MonthlyCharges: float
    TotalCharges: float
    InternetService: str
    PaymentMethod: str


# 2. Create a helper function to write to the CSV
def log_prediction(input_data: dict, prediction: int):
    """Saves the user input, timestamp, and model prediction to a CSV file."""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Check if the file exists so we know whether to write the header row
    file_exists = LOG_FILE.is_file()
    
    # Make a copy of the input so we don't mutate the original request
    log_data = input_data.copy()
    
    # Add our tracking columns
    log_data["timestamp"] = datetime.now(UTC)
    log_data["churn_prediction"] = prediction

    # Append to the CSV file
    with open(LOG_FILE, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_data.keys())
        if not file_exists:
            writer.writeheader()  # Write column names the very first time
        writer.writerow(log_data)

# Health endpoint
@app.get("/health")
def health_check():
    return {"status": "active", "model": "RandomForestClassifier"}

# Prediction endpoint
@app.post("/predict")
def predict_churn(customer: CustomerData):
    try:
        important_features = None
        customer_dict = customer.model_dump()

        with open(feature_file_path, "r") as f:
            important_features = f.read().splitlines()
        
        filtered_data = {key: customer_dict[key] for key in important_features if key in customer_dict}
    
        # Check if we are missing any required features
        if len(filtered_data) != len(important_features):
            missing = set(important_features) - set(filtered_data.keys())
            return {"error": f"Missing required features: {missing}, found dictionary {customer_dict}"}
        
        # Convert the validated JSON payload into a single-row Pandas DataFrame
        filtered_data = pd.DataFrame([filtered_data]) # Use .dict() if on older Pydantic
        
        # Pass the dataframe to our standalone predict.py engine
        probabilities = make_prediction(filtered_data)
        prob = probabilities[0]

        log_prediction(input_data = customer_dict, prediction = int(prob))
        
        # Return the probability AND the default business logic label
        return {
            "churn_probability": round(prob, 4),
            "churn_prediction_default": "Yes" if prob > 0.5 else "No"
        }
        
    except Exception as e:
        # If anything fails (e.g., the encoder crashes), return a proper 500 error
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reload-model")
def reload_model():
    """
    Forces the API to drop the current model from memory 
    and load the latest version from the hard drive.
    """
    global model  # Tell Python we want to modify the global 'model' variable
    
    try:
        # Re-load the model from the same path you defined at the top of your file
        # (Make sure MODEL_PATH matches whatever variable name you used earlier!)
        model = joblib.load(MODEL_PATH) 
        
        # In a real system, you might also reload the 'imp_features.txt' here 
        # if your training script changes the feature columns!
        
        return {"status": "success", "message": "Latest model loaded into memory!"}
        
    except Exception as e:
        return {"status": "error", "message": f"Failed to load model: {str(e)}"}