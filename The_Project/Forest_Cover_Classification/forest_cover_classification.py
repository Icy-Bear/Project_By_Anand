from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from AI_stuff.helper import load_data, preprocess_data, train_model

# Dictionary mapping cover type to image file names
cover_type_dict = {
    1: {"name": "Spruce/Fir", "image": "Spruce_Fir.jpg"},
    2: {"name": "Lodgepole Pine", "image": "Lodgepole_Pine.jpg"},
    3: {"name": "Ponderosa Pine", "image": "Ponderosa_Pine.jpg"},
    4: {"name": "Cottonwood/Willow", "image": "Cottonwood_forest.jpeg"},
    5: {"name": "Aspen", "image": "Aspen.jpg"},
    6: {"name": "Douglas-fir", "image": "Douglas-fir.jpeg"},
    7: {"name": "Krummholz", "image": "Krummholz.jpg"}
}

def get_forest_cover_prediction(feature_values: list) -> Tuple[str, str]:
    """Process the feature values and return the predicted cover type and image."""
    
    # Load data
    df = load_data("Forest_Cover_Classification/assets/forest_covertype.csv")

    # Preprocess data
    target_column = 'Cover_Type'
    X, y = preprocess_data(df, target_column)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = train_model(X_train, y_train, KNeighborsClassifier, n_neighbors=5)

    # Convert input to DataFrame for prediction
    input_data = pd.DataFrame([feature_values], columns=X.columns)

    # Predict cover type
    prediction = model.predict(input_data)[0]
    cover_info = cover_type_dict.get(prediction, {"name": "Unknown", "image": "default.jpg"})
    
    return cover_info['name'], cover_info['image']
