# -1.3598071336738, -0.0727811733098497, 2.53634673796914, 1.37815522427443, -0.338320769942518, 0.462387777762292, 0.239598554061257, 0.0986979012610507, 0.363786969611213, 0.0907941719789316, -0.551599533260813, -0.617800855762348, -0.991389847235408, -0.311169353699879, 1.46817697209427, -0.470400525259478, 0.207971241929242, 0.0257905801985591, 0.403992960255733, 0.251412098239705, -0.018306777944153, 0.277837575558899, -0.110473910188767, 0.0669280749146731, 0.128539358273528, -0.189114843888824, 0.133558376740387, -0.0210530534538215, 0.307940397862009, -0.233093684649867
#sample input
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from AI_stuff.helper import load_data, preprocess_data, split_data, train_model, evaluate_model

def get_credit_card_fraud_prediction(feature_values: list) -> str:
    """Process the feature values and return the prediction for credit card fraud."""
    
    # Load and preprocess the data
    data = load_data("Credit_Card_Fraud_Detection/assets/creditcard.csv")
    legit = data[data.Class == 0]
    fraud = data[data.Class == 1]
    
    # Undersample legitimate transactions to balance the classes
    legit_sample = legit.sample(n=len(fraud), random_state=2)
    data = pd.concat([legit_sample, fraud], axis=0)
    
    # Preprocess data
    X, y = preprocess_data(data, target_column='Class')
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=2)
    
    # Train the model
    model = train_model(X_train, y_train, LogisticRegression)
    
    # Evaluate model performance (optional, can be logged)
    train_acc = evaluate_model(model, X_train, y_train)
    test_acc = evaluate_model(model, X_test, y_test)
    
    # Predict fraud or legit transaction
    input_data = np.array(feature_values).reshape(1, -1)
    prediction = model.predict(input_data)[0]
    
    return "Legitimate transaction" if prediction == 0 else "Fraudulent transaction"

