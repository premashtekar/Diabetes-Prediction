import pandas as pd
import pickle

# Load model
with open("model.pkl", "rb") as f:
    model, feature_cols = pickle.load(f)

# Load test data
test = pd.read_csv("test.csv")

# Fill missing values
for col in test.columns:
    if test[col].dtype == "object":
        test[col].fillna(test[col].mode()[0], inplace=True)
    else:
        test[col].fillna(test[col].median(), inplace=True)

test = pd.get_dummies(test, drop_first=True)

# Align columns
for col in feature_cols:
    if col not in test.columns:
        test[col] = 0

test = test[feature_cols]

# Predict
preds = model.predict_proba(test)[:, 1]

# Load sample submission
sample = pd.read_csv("sample_submission.csv")

# Create submission
sample["diagnosed_diabetes"] = preds
sample.to_csv("submission.csv", index=False)

print("submission.csv generated successfully")
