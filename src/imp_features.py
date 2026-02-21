import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from pathlib import Path

current_dir = Path(__file__).resolve().parent

# Go up one level to the project root, then into the 'models' folder
models_dir = current_dir.parent / "models"
data_dir = current_dir.parent / "data"

feature_file_path = models_dir / "imp_features.txt"
training_data_path = data_dir / "batches" / "batch_0_training.csv"

# Function to detect important features
def permutation_importance(df, target_column):
  # Separate predictors and target
  X = df.drop(target_column, axis = 1)
  y = df[target_column]

  # Split
  X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = 42)

  # Identify categorical columns
  cat_cols = X_train.select_dtypes(include=['object', 'category', 'str']).columns

  # Create a single encoder object for the columns
  enc = OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value = -1)

  # Encode all predictor columns
  X_train[cat_cols] = enc.fit_transform(X_train[cat_cols])
  
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

  # Encode validation columns
  X_val[cat_cols] = enc.transform(X_val[cat_cols])

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

  # Baseline score
  preds = model.predict(X_val)
  baseline_acc_score = accuracy_score(y_val, preds)

  # Permutation run
  val_columns = X_val.columns
  imp_columns = []
  col_names = []

  # Check for each column drop in accuracy
  for column in val_columns:
    temp = X_val[column].copy(deep = True)
    X_val[column] = np.random.permutation(X_val[column].values)

    column_preds = model.predict(X_val)
    column_acc_score = accuracy_score(column_preds, y_val)

    # If difference is greater than 0.01, feature is important
    if baseline_acc_score - column_acc_score > 0.01:
      imp_columns.append({"column": column, "importance_score": baseline_acc_score - column_acc_score})
      col_names.append(column)

    # Assign the original column values back
    X_val[column] = temp

  # Save important features
  with open(feature_file_path, "w") as f:
    f.write("\n".join(col_names))

if __name__ == "__main__":
  training_df = pd.read_csv(training_data_path)
  target_col = training_df.columns[-1]
  permutation_importance(training_df, target_col)