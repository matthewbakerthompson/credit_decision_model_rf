# Credit Decision Model

### **Author**: Matthew Thompson  
### **Date**: 9.4.24  
### **Contact**: [matthewbakerthompson@gmail.com](mailto:matthewbakerthompson@gmail.com)

---

## **Overview**
This script trains a machine learning model to make credit product recommendations and assign a credit limit based on a customer’s creditworthiness. Using a synthetic dataset containing customer attributes like credit score, income, and debt, the model predicts which credit product (e.g., Basic, Gold, Platinum) a customer qualifies for, and provides a recommended credit limit.

---

## **Features**
- **Product Recommendation**: Classifies customers into one of four credit card products: `Basic`, `Standard`, `Gold`, or `Platinum`.
- **Credit Limit Recommendation**: Provides a credit limit based on the customer's predicted product category and their income level.
- **Hyperparameter Tuning**: Uses `GridSearchCV` for hyperparameter tuning to optimize the model’s performance and prevent overfitting.
- **Feature Importance**: Outputs the importance of features used in decision-making to ensure model transparency.
- **Model Evaluation**: Provides a detailed classification report, confusion matrix, and accuracy score for the trained model.

---

## **Inputs**
- **Dataset**: The model uses the `enhanced_credit_risk_data.csv` dataset, which includes customer credit information (e.g., `Credit_Score`, `Income`, `Debt`, etc.).
  - **Customer Attributes**: Age, Gender, Region, Education Level, Employment Status, Income, Credit Score, Debt Metrics, and more.
  - **Generated with**: A synthetic data generation script that uses the `Faker` library and custom logic.

---

## **Outputs**
- **Model Metrics**: Prints key evaluation metrics including:
  - **Confusion Matrix**: To assess classification performance.
  - **Classification Report**: Provides precision, recall, and F1-score for each product class.
  - **Accuracy**: Overall accuracy of the model on the test data.
- **Predicted Product and Credit Limit**: Shows the recommended product and credit limit for each test customer.

---

## **Model Architecture**
- **Random Forest Classifier**: 
  - A robust, ensemble learning algorithm that fits multiple decision trees on subsets of the dataset and averages the results to improve accuracy and control overfitting.
  - **Hyperparameter Tuning**:
    - `max_depth`: The maximum depth of each decision tree.
    - `min_samples_split`: The minimum number of samples required to split a node.
    - `n_estimators`: The number of trees in the forest.
    
---

## **Workflow**

### **Step 1: Preprocess Data**
- **Categorical Variables**: Encode `Gender`, `Region`, `Education Level`, `Employment Status`, `Housing`, and `Payment History` using `LabelEncoder`.
- **Numerical Variables**: Scale all numerical features (e.g., `Income`, `Credit_Score`, `Existing_Debt`, etc.) using `StandardScaler`.

### **Step 2: Train-Test Split**
- Split the dataset into 80% training and 20% testing subsets using `train_test_split()`.

### **Step 3: Train Model**
- Train a `RandomForestClassifier` using the training dataset.
- Perform hyperparameter tuning using `GridSearchCV` to optimize the model parameters.

### **Step 4: Evaluate Model**
- Generate the following:
  - **Confusion Matrix**: To assess how well the model classifies customers into product categories.
  - **Classification Report**: Detailing precision, recall, and F1-score for each class.
  - **Accuracy**: Overall accuracy of the model on the test data.
  
### **Step 5: Feature Importance**
- The script computes and prints the importance of each feature to explain the model’s decision-making.

### **Step 6: Predict Credit Product and Limit**
- Based on a customer’s financial profile, the model predicts which credit card product is suitable and assigns a credit limit based on their income and product.

---

## **Model Governance**
### **Fairness and Transparency**
- **Feature Importance**: Transparency in decision-making is ensured by analyzing feature importance to determine which factors contribute most to model predictions.
  
### **Overfitting Prevention**
- **Hyperparameter Tuning**: Using cross-validation with `GridSearchCV` helps avoid overfitting by tuning the model parameters.
  
### **Performance Tracking**
- **Evaluation Metrics**: Metrics such as accuracy, precision, recall, and F1-score are logged to ensure model performance remains acceptable.
  
### **Auditability**
- **Model Traceability**: Each step of the process (data preprocessing, training, and evaluation) is documented and can be reproduced for audit purposes.

---

## **Key Dependencies**
- `pandas`: For data manipulation.
- `numpy`: For numerical operations.
- `scikit-learn`: For building the machine learning model (`RandomForestClassifier`, `GridSearchCV`, and utilities like `train_test_split` and `StandardScaler`).
- `matplotlib`: (Optional) For visualizing results.

---

## **Usage**

1. Clone the repository or download the script.
2. Make sure the `enhanced_credit_risk_data.csv` file is in the same directory as the script.
3. Install the required dependencies:
    ```bash
    pip install pandas numpy scikit-learn
    ```
4. Run the script:
    ```bash
    python credit_decision_model.py
    ```

---

## **Example Output**

### **Credit Product Recommendation Accuracy**: 0.9997

### **Classification Report**:
| Product        | Precision | Recall | F1-Score | Support |
|----------------|-----------|--------|----------|---------|
| Basic Card     | 1.00      | 1.00   | 1.00     | 508     |
| Gold Card      | 1.00      | 1.00   | 1.00     | 1064    |
| Platinum Card  | 1.00      | 1.00   | 1.00     | 651     |
| Standard Card  | 1.00      | 1.00   | 1.00     | 777     |

### Confusion Matrix:

|               | Basic Card | Gold Card | Platinum Card | Standard Card |
|---------------|------------|-----------|---------------|---------------|
| **Basic Card**     | 508        | 0         | 0             | 0             |
| **Gold Card**      | 0          | 1064      | 0             | 0             |
| **Platinum Card**  | 0          | 0         | 651           | 0             |
| **Standard Card**  | 1          | 0         | 0             | 776           |


### **Sample Predictions**:
| Credit_Score | Predicted_Product | Income    | Credit_Limit |
|--------------|-------------------|----------|--------------|
| 810.88       | Platinum Card      | 20000.00 | 10000        |
| 723.01       | Gold Card          | 105959.13| 5000         |
| 668.12       | Standard Card      | 30986.78 | 1000         |
| 607.52       | Standard Card      | 60000.00 | 1000         |
| 815.47       | Platinum Card      | 20000.00 | 10000        |


---

## **Future Enhancements**
- **Additional Products**: Expand to more complex product recommendations with different tiers or credit requirements.
- **Feature Engineering**: Incorporate additional behavioral or historical features (e.g., spending patterns) to refine the recommendation system.
- **Explainability**: Integrate SHAP (SHapley Additive exPlanations) to further interpret and explain model decisions.

---

## **License**
This project is licensed under the MIT License.
