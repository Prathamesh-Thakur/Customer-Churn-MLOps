import joblib
import pandas as pd
from pathlib import Path

# Find exactly where predict.py is located on the hard drive (the 'src' folder)
current_dir = Path(__file__).resolve().parent

# Go up one level to the project root, then into the 'models' folder
models_dir = current_dir.parent / "models"

# 3. Create the full, absolute path to the files
feature_file_path = models_dir / "imp_features.txt"
model_file_path = models_dir / "training_model.joblib"
encoder_file_path = models_dir / "encoder.joblib"

# We load the model and encoder outside the function. 
# This way, when FastAPI is running, it loads them into memory ONCE at startup, 
# rather than reading from the hard drive every single time a user makes a request.

try:
    model = joblib.load(model_file_path)
    encoder = joblib.load(encoder_file_path)
except FileNotFoundError:
    print("Warning: Model or Encoder not found. Run train.py first.")

# Get important features from permutation
important_features = None
with open(feature_file_path, "r") as f:
    important_features = f.read().splitlines()


def make_prediction(raw_df: pd.DataFrame) -> list:
    """
    Takes a raw pandas DataFrame (which might have extra columns like 'Gender' 
    or 'PaymentMethod'), filters it, encodes it, and returns Churn probabilities.
    """
    # Drop any columns the model doesn't need
    # We use .copy() to avoid SettingWithCopy warnings in Pandas
    df_clean = raw_df[important_features].copy()
    
    # 2. Encode: Transform categorical text into numbers using our saved encoder
    cat_cols = raw_df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    df_clean[cat_cols] = encoder.transform(df_clean[cat_cols])
    
    # 3. Predict: Get the probability of Churn (Class 1)
    probabilities = model.predict_proba(df_clean)[:, 1]
    
    return probabilities.tolist()