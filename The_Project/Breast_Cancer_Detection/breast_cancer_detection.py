# Sample Input
# -0.23717126, -0.64487029, -0.11382239, -0.57427777, -0.60294971, 1.0897546 ,  0.91543814,  0.41448279,  0.09311633,  1.78465117, 2.11520208,  0.28454765, -0.31910982,  0.2980991 ,  0.01968238, -0.47096352,  0.45757106,  0.28733283, -0.23125455,  0.26417944, 0.66325388,  0.12170193,  0.42656325,  0.36885508,  0.02065602, 1.39513782,  2.0973271 ,  2.01276347,  0.61938913,  2.9421769 , 3.15970842
from typing import Tuple
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from AI_stuff.helper import load_data, preprocess_data, split_data, train_model, evaluate_model

def get_breast_cancer_prediction(feature_values: list) -> Tuple[str]:
    """Process the feature values and return the predicted cancer type."""
    
    # Load the dataset
    breast = load_data("Breast_Cancer_Detection/assets/breast_cancer.csv")

    # Drop unnecessary columns
    breast.drop('Unnamed: 32', axis=1, inplace=True)

    # Preprocess data
    X, y = preprocess_data(breast, target_column='diagnosis')

    # Encode categorical labels
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train the logistic regression model
    model = train_model(X_train, y_train, LogisticRegression)

    # Scale the input data
    input_data = pd.DataFrame([feature_values], columns=X.columns)
    input_data = scaler.transform(input_data)

    # Predict cancer type
    prediction = model.predict(input_data)[0]
    result = "Cancerous" if prediction == 1 else "Not Cancerous"
    
    return result

