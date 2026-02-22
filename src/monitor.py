import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

import joblib

current_dir = Path(__file__).resolve().parent

# Go up one level to the project root, then into the 'models' folder
models_dir = current_dir.parent / "models"
data_dir = current_dir.parent / "data" / "batches"

# Create the full, absolute path to the files
feature_file_path = models_dir / "imp_features.txt"
model_file_path = models_dir / "training_model.joblib"
encoder_file_path = models_dir / "encoder.joblib"
val_probs_path = models_dir / "validation_probabilities.npy"
psi_scores_path = models_dir / "PSI_scores.txt"

# Data files
training_data_path = data_dir / "batch_0_training.csv"
# Use this path and disable checking if you just want to check out the working of the DAG
# prod_data_path = data_dir / "batch_1_production.csv"

# Points to the file generated through live input
prod_data_path = data_dir / "live_inference_logs.csv"

def calc_psi_score(expected, actual, col_type = "numeric"):
  """Function to calculate the Population Stability Index"""
  # Empty series objects
  train_val_counts = pd.Series()
  prod_val_counts = pd.Series()

  if col_type not in ['object', 'category']: # If column type is numeric
    if expected.nunique() > 20: # Check if the column is continuous or discrete encoded
      out, training_bins = pd.qcut(expected, 10, duplicates = 'drop', retbins = True) # Bin the data

      cleaned_bins = np.round(training_bins, 3) # Round the bin interval limits for cleanliness

      # Add -inf and inf as extremes so that outliers are added to a bin
      cleaned_bins[0] = -np.inf
      cleaned_bins[-1] = np.inf

      expected = pd.cut(expected, cleaned_bins) # Bin the training data column

      actual = pd.cut(actual, cleaned_bins) # Bin the production data column

  # Counts of every unique bin interval
  train_val_counts = expected.value_counts(normalize = True, dropna = False)
  prod_val_counts = actual.value_counts(normalize = True, dropna = False)

  # Convert index to string 
  train_val_counts.index = train_val_counts.index.astype(str)
  prod_val_counts.index = prod_val_counts.index.astype(str)

  # Get all features if some are missed
  all_categories = list(set(train_val_counts.index) | set(prod_val_counts.index))

  # Align both indexes
  train_val_counts = train_val_counts.reindex(all_categories, fill_value = 0)
  prod_val_counts = prod_val_counts.reindex(all_categories, fill_value = 0)

  # Renormalize the value counts
  prod_val_counts = prod_val_counts / prod_val_counts.sum()

  epsilon = 1e-4 # Small value to avoid division by zero error

  train_val_counts = train_val_counts + epsilon
  prod_val_counts = prod_val_counts + epsilon

  # Calculate the psi score
  psi_values = (prod_val_counts - train_val_counts) * np.log(prod_val_counts / train_val_counts)
  psi_total = psi_values.sum()

  return round(psi_total, 4)

def prod_data_probs(prod_X):
  """Function to encode the production data"""
  # Load the model and encoder
  training_model = joblib.load(model_file_path)
  encoder = joblib.load(encoder_file_path)

  # Get the categorical columns and encode them
  cat_cols = prod_X.select_dtypes(include=['object', 'category']).columns
  prod_X[cat_cols] = encoder.transform(prod_X[cat_cols])

  # Calculate the expected probabilities for the production data
  prod_probs = training_model.predict_proba(prod_X)[:, 1]

  return prod_probs


def run_drift_report():
  """Generate the drift report"""
  important_features = None

  # Get the important features
  with open(feature_file_path, "r") as f:
    important_features = f.read().splitlines()
   
  # Load the training and production data
  df_training = pd.read_csv(training_data_path)
  
  # Safely load the live logs
  try:
      df_prod = pd.read_csv(prod_data_path)
  except FileNotFoundError:
      print("No live inference logs found yet. Skipping drift detection.")
      return False

  # Check for minimum sample size (e.g., at least 50 predictions)
  min_samples_for_psi = 50
  if len(df_prod) < min_samples_for_psi:
    print(f"Only {len(df_prod)} live predictions logged. Waiting for at least {min_samples_for_psi} to calculate statistically significant PSI.")
    return False

  psi_scores = ""

  psi_scores += f"------- DRIFT REPORT {datetime.now(timezone.utc)} ---------\n"
  psi_scores += "=== COVARIATE SHIFT REPORT (INPUTS) ===\n"

  drift_detected = False

  # Check for all important feature columns
  for col in important_features:
    psi_score = calc_psi_score(df_training[col], df_prod[col], df_prod[col].dtype)
    # Based on threshold, save the outcome in the report
    if psi_score < 0.1:
      psi_scores += f"[{col}] GREEN - PSI: {psi_score:.4f}\n"
    elif 0.1 <= psi_score < 0.2:
      psi_scores += f"[{col}] YELLOW - PSI: {psi_score:.4f}\n"
    else:
      psi_scores += f"[{col}] RED - PSI: {psi_score:.4f} (DRIFT DETECTED!)\n"
      drift_detected = True
  
  psi_scores += "\n=== PRIOR PROBABILITY SHIFT REPORT (OUTPUTS) ===\n"
  
  df_prod_X = df_prod.drop("Churn", axis = 1)
  df_prod_X = df_prod_X[important_features]

  prod_probs = prod_data_probs(df_prod_X)

  val_probs = np.load(val_probs_path)

  psi_score = calc_psi_score(pd.Series(val_probs), pd.Series(prod_probs))
  if psi_score > 0.2:
    psi_scores += f"[Probability] RED - PSI: {psi_score:.4f} (DRIFT DETECTED!)\n"
    drift_detected = True
  else:
    psi_scores += f"[Probability] GREEN - PSI: {psi_score:.4f}\n\n"
  
  # Write the report
  with open(psi_scores_path, "a") as f:
      f.write(psi_scores)
  
  return drift_detected