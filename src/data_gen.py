import os
import kagglehub
import random

import numpy as np
import pandas as pd

# Set download location to desired directory
new_cache_dir = "../data/raw"
os.environ["KAGGLEHUB_CACHE"] = new_cache_dir

# Download the dataset from kagglehub
path = kagglehub.dataset_download("blastchar/telco-customer-churn")

# List all files in the directory
for file_name in os.listdir(path):
    print(file_name)

# Read the file
df = pd.read_csv(path + "/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Basic cleaning
df = df.drop('customerID', axis = 1)

# Replace empty strings with nulls, drop them and convert to float
df.replace(r"^\s*$", np.nan, regex=True, inplace=True)
df['TotalCharges'].dropna(axis = 0, inplace = True)
df['TotalCharges'] = df['TotalCharges'].astype(float)

# Baseline Data
df_training = df.sample(frac = 0.5, random_state = 42).reset_index(drop = True) # Separate out half the data for the baseline
df_training.to_csv("../data/batches/batch_0_training.csv", index = False) # Save it

df_rest = df.drop(df_training.index) # Remove the baseline data from the original data

# ------------------------- BATCH CREATION -----------------------------

# Feature shift

# Get the filtered data for simulating age
high_risk_customers = df_rest[df_rest["Contract"] == "Month-to-month"]

# "Manufacture" a larger dataset by sampling with replacement
df_contract_shift = pd.concat([df_rest, high_risk_customers, high_risk_customers], axis = 0)

df_contract_shift.to_csv("../data/batches/batch_1_production.csv", index = False)

# Concept Shift

# Create a deep copy
df_cov_shift = df_rest.copy(deep = True)

# Change 50% of rows to simulate drift
temp = df_cov_shift[df_cov_shift["PaymentMethod"] == "Credit card (automatic)"]
index_count = int(len(temp) * 0.5)
indexes_changed = random.sample(list(temp.index), k = index_count)

df_cov_shift.loc[indexes_changed, "Churn"] = "Yes"

df_cov_shift.to_csv("../data/batches/batch_2_production.csv", index = False) # Export to csv