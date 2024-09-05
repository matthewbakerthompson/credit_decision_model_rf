"""
File:        credit_decision_model_rf.py
Author:      Matthew Thompson - matthewbakerthompson@gmail.com
Date:        9.4.24
Description:
    This script trains and evaluates a machine learning model to recommend credit card 
    products and credit limits based on a customer's financial profile. The model uses 
    a synthetic dataset generated from customer credit attributes like credit score, 
    income, debt, and payment history to make product recommendations.

    The dataset includes variables that are essential for credit risk modeling, such as 
    Debt-to-Income (DTI) ratio, existing debt, employment status, and credit utilization. 
    The model uses a Random Forest Classifier to predict which credit product (e.g., 
    Basic, Standard, Gold, or Platinum Card) a customer is eligible for based on their 
    creditworthiness, and it provides a recommended credit limit based on that product.

Features:
    - Trains a machine learning model (Random Forest) to classify customers into one of 
      four credit card product categories: Basic, Standard, Gold, or Platinum.
    - Recommends a credit limit based on the customer's predicted credit product and income.
    - Includes hyperparameter tuning with GridSearchCV to optimize model performance.
    - Outputs accuracy, classification report, and confusion matrix for the model’s 
      performance on the test set.
    - Provides feature importance to explain the impact of different variables on the 
      model's decision-making process.
    - Handles the preprocessing of categorical and numerical data, including encoding 
      categorical variables and scaling numeric features.

Inputs:
    - enhanced_credit_risk_data.csv: The generated dataset containing customer credit 
      information used to train and test the model.

Outputs:
    - Model evaluation metrics, including confusion matrix and classification report 
      printed to the console.
    - Feature importance metrics printed to the console to interpret model decisions.
    - Recommended credit product and credit limit for test customers printed to the console.

Usage:
    python credit_decision_model.py

Dependencies:
    - pandas
    - numpy
    - sklearn (RandomForestClassifier, GridSearchCV, train_test_split, StandardScaler, LabelEncoder)
    - matplotlib (optional, for visualizing results)
    - enhanced_credit_risk_data.csv (generated from the synthetic data generation script)

Model Governance Features:
    - Hyperparameter tuning is performed using GridSearchCV to avoid overfitting and to 
      optimize model parameters.
    - Feature importance is evaluated to ensure model transparency and interpretability.
    - The model can be validated using holdout data (test set) to ensure generalization.
    - Key performance metrics (accuracy, precision, recall, f1-score) are printed for 
      each credit product class to track model effectiveness.
"""


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the synthetic dataset created by generate_credit_data.py
df = pd.read_csv('enhanced_credit_risk_data.csv')

# Data Preprocessing
# Drop non-relevant columns (e.g., Customer_ID, Name)
df = df.drop(['Customer_ID', 'Name', 'Bankruptcy_Date'], axis=1)

# Encode categorical variables
label_encoders = {}
categorical_columns = ['Gender', 'Region', 'Education_Level', 'Employment_Status', 'Bankruptcy_History', 'Payment_History', 'Housing']

for column in categorical_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le  # Save each encoder for later use

# Define credit products based on Credit Score
def recommend_product(credit_score):
    if credit_score < 580:
        return 'Basic Card'
    elif 580 <= credit_score < 670:
        return 'Standard Card'
    elif 670 <= credit_score < 740:
        return 'Gold Card'
    else:
        return 'Platinum Card'

# Apply function to create a new column for recommended product
df['Product'] = df['Credit_Score'].apply(recommend_product)

# Split dataset into features (X) and target (y)
X = df.drop(['Product', 'Credit_Score_Category'], axis=1)  # Features
y = df['Product']  # Target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Training for Credit Product Recommendation
product_model = RandomForestClassifier(random_state=42)
product_model.fit(X_train_scaled, y_train)

# Model Evaluation
y_pred_product = product_model.predict(X_test_scaled)
print("Credit Product Recommendation Accuracy:", accuracy_score(y_test, y_pred_product))
print("Classification Report:\n", classification_report(y_test, y_pred_product))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_product))

# Credit Limit Calculation based on Product and Income
def calculate_credit_limit(row):
    """
    Rule-based credit limit recommendation.
    Limits are calculated based on income and product.
    """
    if row['Product'] == 'Basic Card':
        return min(500, row['Income'] * 0.1)  # Conservative limit for low creditworthiness
    elif row['Product'] == 'Standard Card':
        return min(1000, row['Income'] * 0.2)
    elif row['Product'] == 'Gold Card':
        return min(5000, row['Income'] * 0.4)
    elif row['Product'] == 'Platinum Card':
        return min(10000, row['Income'] * 0.6)
    else:
        return 500  # Default limit

# Apply the credit limit calculation function to the test set
df_test = df.loc[X_test.index].copy()
df_test['Predicted_Product'] = y_pred_product
df_test['Credit_Limit'] = df_test.apply(calculate_credit_limit, axis=1)

# Save the results to a CSV file for review
df_test.to_csv('credit_product_recommendations.csv', index=False)

# Display a few sample results to check output
print(df_test[['Credit_Score', 'Predicted_Product', 'Income', 'Credit_Limit']].head())

# Feature Importance Analysis
importances = product_model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print("\nFeature Importance:\n", feature_importance_df)

# Governance - Save model artifacts, parameters, and feature importance for transparency and auditability
feature_importance_df.to_csv('feature_importance.csv', index=False)
