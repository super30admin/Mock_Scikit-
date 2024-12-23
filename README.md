# Mock_Scikit

1 problem 1 : https://tinyurl.com/scikitMockInterviewQuestion

# Customer Churn Prediction Model

This repository contains a machine learning pipeline to predict customer churn for a telecommunications company using the dataset `Telco-Customer-Churn.csv`. The aim is to help the company identify customers likely to churn and enable targeted retention efforts.

---

## Project Workflow

1. **Load and Preprocess the Dataset**
2. **Split the Dataset into Training and Testing Sets**
3. **Feature Scaling**
4. **Model Selection and Training**
5. **Model Evaluation**
6. **Predict Churn for New Data**

---

## Code Overview

### Loading the Dataset

```python
import pandas as pd

# Load dataset
data = pd.read_csv("Telco-Customer-Churn.csv")

# Explore data
print(data.info())
print(data.describe())
# Handle missing values
data.fillna(data.mean(), inplace=True)
data.fillna(data.mode().iloc[0], inplace=True)

# Encode categorical variables
data["Churn"] = data["Churn"].map({'Yes': 1, 'No': 0})

from sklearn.model_selection import train_test_split

# Separate features and target
X = data.drop(columns=["Churn"])
y = data["Churn"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.preprocessing import StandardScaler

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
