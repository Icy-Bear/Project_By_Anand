from typing import List
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from AI_stuff.helper import load_data, split_data, train_model

def get_crop_recommendation(features: List[float]) -> str:
    """Process the feature values and return the recommended crop."""
    
    # Load and prepare crop data
    file_path = "Crop_Recommendation/assets/Crop_recommendation.csv"  # Update file path if needed
    crop = load_data(file_path)
    
    # Ensure 'label' column is present
    if 'label' not in crop.columns:
        raise ValueError("The 'label' column is missing from the dataset.")
    
    # Map crop labels to numeric values
    crop['crop_num'] = crop['label'].map({
        'rice': 1, 'maize': 2, 'jute': 3, 'cotton': 4, 'coconut': 5, 'papaya': 6, 'orange': 7,
        'apple': 8, 'muskmelon': 9, 'watermelon': 10, 'grapes': 11, 'mango': 12, 'banana': 13,
        'pomegranate': 14, 'lentil': 15, 'blackgram': 16, 'mungbean': 17, 'mothbeans': 18,
        'pigeonpeas': 19, 'kidneybeans': 20, 'chickpea': 21, 'coffee': 22
    })
    
    if 'crop_num' not in crop.columns:
        raise ValueError("The 'crop_num' column is missing after mapping.")

    crop.drop(['label'], axis=1, inplace=True)
    X = crop.drop(['crop_num'], axis=1)
    y = crop['crop_num']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train the RandomForest model
    model = train_model(X_train, y_train, RandomForestClassifier)
    
    # Transform the input data
    features_array = np.array([features])
    transformed_features = scaler.transform(features_array)
    
    # Predict the crop
    prediction = model.predict(transformed_features)[0]
    
    # Inverse mapping
    CROP_DICT_INV = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                     8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                     14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                     19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}
    
    return CROP_DICT_INV.get(prediction, "Unknown")

