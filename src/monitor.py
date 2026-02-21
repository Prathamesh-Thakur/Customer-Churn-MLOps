import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

import joblib

current_dir = Path(__file__).resolve().parent

# Go up one level to the project root, then into the 'models' folder
models_dir = current_dir.parent / "models"
data_dir = current_dir.parent / "data" / "batches"

# 3. Create the full, absolute path to the files
feature_file_path = models_dir / "imp_features.txt"
model_file_path = models_dir / "training_model.joblib"
encoder_file_path = models_dir / "encoder.joblib"
val_probs_path = models_dir / "validation_probabilities.npy"
psi_scores_path = models_dir / "PSI_scores.txt"

training_data_path = data_dir / "batch_0_training.csv"
prod_data_path = data_dir / "batch_1_production.csv"

def calc_psi_score(expected, actual, col_type = "numeric"):
  train_val_counts = pd.Series()
  prod_val_counts = pd.Series()

  if col_type not in ['object', 'category']:
    if expected.nunique() > 20:
      out, training_bins = pd.qcut(expected, 10, duplicates = 'drop', retbins = True)

      cleaned_bins = np.round(training_bins, 3)

      cleaned_bins[0] = -np.inf
      cleaned_bins[-1] = np.inf

      expected = pd.cut(expected, cleaned_bins)

      actual = pd.cut(actual, cleaned_bins)

  train_val_counts = expected.value_counts(normalize = True, dropna = False)
  prod_val_counts = actual.value_counts(normalize = True, dropna = False)

  train_val_counts.index = train_val_counts.index.astype(str)
  prod_val_counts.index = prod_val_counts.index.astype(str)

  all_categories = list(set(train_val_counts.index) | set(prod_val_counts.index))

  train_val_counts = train_val_counts.reindex(all_categories, fill_value = 0)
  prod_val_counts = prod_val_counts.reindex(all_categories, fill_value = 0)

  prod_val_counts = prod_val_counts / prod_val_counts.sum()

  epsilon = 1e-4

  train_val_counts = train_val_counts + epsilon
  prod_val_counts = prod_val_counts + epsilon

  psi_values = (prod_val_counts - train_val_counts) * np.log(prod_val_counts / train_val_counts)
  psi_total = psi_values.sum()

  return round(psi_total, 4)


# def generate_alert(psi_score, name):
#     """Helper function to print/log the Green/Yellow/Red status"""
    # if psi_score < 0.1:
    #     print(f"[{name}] GREEN - PSI: {psi_score:.4f}")
    # elif 0.1 <= psi_score < 0.2:
    #     print(f"[{name}] YELLOW - PSI: {psi_score:.4f}")
    # else:
    #     print(f"[{name}] RED - PSI: {psi_score:.4f} (DRIFT DETECTED!)")


def prod_data_probs(prod_X):
  training_model = joblib.load(model_file_path)

  encoder = joblib.load(encoder_file_path)

  cat_cols = prod_X.select_dtypes(include=['object', 'category']).columns

  prod_X[cat_cols] = encoder.transform(prod_X[cat_cols])

  prod_probs = training_model.predict_proba(prod_X)[:, 1]

  return prod_probs


def run_drift_report():
  important_features = None

  with open(feature_file_path, "r") as f:
    important_features = f.read().splitlines()
   
  df_training = pd.read_csv(training_data_path)

  df_prod = pd.read_csv(prod_data_path)

  psi_scores = ""

  psi_scores += f"------- DRIFT REPORT {datetime.now(timezone.utc)} ---------\n"
  psi_scores += "=== COVARIATE SHIFT REPORT (INPUTS) ===\n"

  drift_detected = False

  for col in important_features:
    psi_score = calc_psi_score(df_training[col], df_prod[col], df_prod[col].dtype)
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
  
  with open(psi_scores_path, "a") as f:
      f.write(psi_scores)
  
  return drift_detected