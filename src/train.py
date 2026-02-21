import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import pickle
import joblib

from pathlib import Path

current_dir = Path(__file__).resolve().parent

# Go up one level to the project root, then into the 'models' folder
models_dir = current_dir.parent / "models"
data_dir = current_dir.parent / "data" 

# 3. Create the full, absolute path to the files
feature_file_path = models_dir / "imp_features.txt"
model_file_path = models_dir / "training_model.joblib"
encoder_file_path = models_dir / "encoder.joblib"
target_encoder_path = models_dir / "target_mappings.pkl"
val_probs_path = models_dir / "validation_probabilities.npy"

training_data_path = data_dir / "batches" / "batch_0_training.csv"
prod_data_path = data_dir / "live_inference_logs.csv"

# Function to train model
def training(target_column):
  # Read the important features calculated
  important_features = None
  with open(feature_file_path, "r") as f:
    important_features = f.read().splitlines()

  # Separate predictors and target
  df = pd.read_csv(training_data_path)
  X = df.drop(target_column, axis = 1)
  X = X[important_features]
  y = df[target_column]

  df_live = pd.DataFrame()

  if prod_data_path.is_file():
    print("Live inference logs found! Processing new data...")
    df_live = pd.read_csv(prod_data_path)
    
    # --- SIMULATING GROUND TRUTH ---
    # Here, we mock the 'Churn' label so the pipeline works. Let's assume the model's prediction was right.
    if 'churn_prediction' in df_live.columns:
        # Map the 1/0 prediction back to Yes/No (or whatever format your data uses)
        df_live[target_column] = df_live['churn_prediction'].map({1: 'Yes', 0: 'No'})
        
    # Drop the tracking columns FastAPI added so the schemas match perfectly
    cols_to_drop = ['timestamp', 'churn_prediction']
    df_live = df_live.drop(columns=[c for c in cols_to_drop if c in df_live.columns])

    X = pd.concat([X, df_live.reindex(columns = X.columns)], ignore_index=True)
    y = pd.concat([y, df_live[target_column]], ignore_index=True)
  
  WINDOW_SIZE = 5000 
  if len(X) > WINDOW_SIZE:
    X = X.tail(WINDOW_SIZE)
    print(f"Data merged. Training on sliding window of {len(df)} records.")
  
  else:
    print("No live logs found. Training on historical data only.")
  
  # Split
  X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = 42)

  # Identify categorical columns
  cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

  # Create a single encoder object for the columns
  feature_enc = OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value = -1)

  # Encode all predictor columns
  X_train[cat_cols] = feature_enc.fit_transform(X_train[cat_cols])

  # Save the predictor encoders
  joblib.dump(feature_enc, encoder_file_path)
  
  # Dictionary for target mappings
  target_mappings = {}

  target_is_categorical = False
  # Encode target if not numeric
  if y.dtype == 'object':
    target_is_categorical = True
    enc = LabelEncoder()
    enc.fit(y_train)
    y_train = enc.transform(y_train)
    target_mappings[target_column] = enc

  with open(target_encoder_path, 'wb') as f:
    pickle.dump(target_mappings, f)

  # Encode validation columns
  X_val[cat_cols] = feature_enc.transform(X_val[cat_cols])

  if target_is_categorical:
    enc = target_mappings[target_column]
    y_val = enc.transform(y_val)

  # Model training and inference for basline score
  model = RandomForestClassifier(
    n_estimators = 150,
    max_depth = 7,
    random_state = 42
  )

  # Train the model
  model.fit(X_train, y_train)

  joblib.dump(model, model_file_path)

  # Baseline score
  preds = model.predict(X_val)
  acc_score = accuracy_score(y_val, preds)
  print("Accuracy score:", acc_score)

  # Get probabilities for psi calculation
  probs = model.predict_proba(X_val)[:, 1]

  # Save it for future use
  np.save(val_probs_path, probs)

training("Churn")