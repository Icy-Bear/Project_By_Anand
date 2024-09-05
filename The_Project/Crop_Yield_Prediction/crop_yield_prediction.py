from typing import List, Union
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from AI_stuff.helper import load_data, split_data, train_model

def get_crop_yield_prediction(features: List[Union[float, str]]) -> float:
    """Process the feature values and return the predicted crop yield."""

    # Load and prepare crop yield data
    file_path = "Crop_Yield_Prediction/assets/yield_df.csv"  # Update file path if needed
    crop_yield_data = load_data(file_path)
    
    # Drop unnecessary columns and handle duplicates
    if 'Unnamed: 0' in crop_yield_data.columns:
        crop_yield_data.drop('Unnamed: 0', axis=1, inplace=True)
    crop_yield_data.drop_duplicates(inplace=True)
    
    # Filter out non-numeric values in the 'average_rain_fall_mm_per_year' column
    def isStr(obj):
        try:
            float(obj)
            return False
        except ValueError:
            return True
    
    to_drop = crop_yield_data[crop_yield_data['average_rain_fall_mm_per_year'].apply(isStr)].index
    crop_yield_data.drop(to_drop, inplace=True)
    crop_yield_data['average_rain_fall_mm_per_year'] = crop_yield_data['average_rain_fall_mm_per_year'].astype(np.float64)
    
    # Select relevant columns
    columns = ['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp', 'Area', 'Item', 'hg/ha_yield']
    crop_yield_data = crop_yield_data[columns]
    X = crop_yield_data.iloc[:, :-1]
    y = crop_yield_data.iloc[:, -1]
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
    
    # Preprocess the features
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.compose import ColumnTransformer
    
    ohe = OneHotEncoder(drop='first')
    scaler = StandardScaler()
    
    preprocesser = ColumnTransformer(
        transformers=[
            ('StandardScale', scaler, [0, 1, 2, 3]),  
            ('OHE', ohe, [4, 5]),                    
        ],
        remainder='passthrough'
    )
    
    X_train_processed = preprocesser.fit_transform(X_train)
    X_test_processed = preprocesser.transform(X_test)
    
    # Train the DecisionTree model
    model = train_model(X_train_processed, y_train, DecisionTreeRegressor)
    
    # Transform the input features
    features_array = np.array([features], dtype=object)
    transformed_features = preprocesser.transform(features_array)
    
    # Predict the crop yield
    predicted_yield = model.predict(transformed_features)[0]
    
    return predicted_yield

