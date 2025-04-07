# Customer Churn Prediction - End-to-End Machine Learning Project

## Project Overview
This project predicts customer churn for a telecom company using Machine Learning and Deep Learning techniques. The project covers a complete pipeline from data preprocessing to model evaluation. A variety of models are trained and compared, including Random Forest, XGBoost, Gradient Boosting, Neural Network, and a Stacking Classifier.

## Dataset
- The dataset includes telecom customer data such as service usage patterns, location, and customer details.
- Feature engineering includes creating a new tenure_days column from registration date.
- Skewed numerical features are transformed using np.log1p for better model performance.
- Class imbalance is addressed using ADASYN (Adaptive Synthetic Sampling).

## Data Preprocessing
- Datetime Parsing: date_of_registration converted to datetime and used to derive tenure_days.
- Feature Selection: Irrelevant columns like customer_id, pincode, and date_of_registration are dropped.
- Label Encoding: Categorical columns (telecom_partner, gender, state, city) are label encoded.
- Skewness Reduction: log1p transformation is applied to features like calls_made, sms_sent, data_used, and tenure_days.

## Models Used
- Random Forest
- Gradient Boosting Classifier
- XGBoost Classifier
- Neural Network (TensorFlow/Keras)
- Stacking Classifier

## Performance Evaluation
Each model is evaluated based on:
- Accuracy Score
- ROC-AUC Score
- Classification Report (Precision, Recall, F1-Score)
- Confusion Matrix

## Hyperparameter Tuning
- Grid search (GridSearchCV) is applied for tuning Random Forest and XGBoost models.
- Final model selection is based on balanced accuracy and ROC-AUC.

## How to Use
1. Clone this repository:
   git clone https://github.com/yourusername/customer-churn-prediction.git
2. Install dependencies:
   pip install -r requirements.txt
3. Run the Jupyter Notebook or script:
   jupyter notebook Customer_Churn_Prediction.ipynb
4. Train models and evaluate results.

## Future Improvements
- Deploy the best model using Flask/FastAPI.
- Integrate SHAP for Explainable AI (XAI) to understand feature impact.
- Experiment with LSTM or Transformers for sequential churn prediction.

## Connect
If you find this project useful, give it a star and follow me on www.linkedin.com/in/vidhyarajan-mohan-6bb572137.
