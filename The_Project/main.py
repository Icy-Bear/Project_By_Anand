import pandas as pd
import base64
import streamlit as st
from Forest_Cover_Classification.forest_cover_classification import get_forest_cover_prediction
from Crop_Recommendation.crop_recommendation import get_crop_recommendation
from AI_stuff.helper import load_data
from Crop_Yield_Prediction.crop_yield_prediction import get_crop_yield_prediction
from Credit_Card_Fraud_Detection.credit_card_fraud_detection import get_credit_card_fraud_prediction 
from Breast_Cancer_Detection.breast_cancer_detection import get_breast_cancer_prediction



def create_breast_cancer_ui():
    """Create the Streamlit UI for breast cancer detection."""
    
    st.subheader("Breast Cancer Detection")

    # Add custom CSS for controlling image height
    st.markdown(
        """
        <style>
        .full-width-image {
            width: 100%;
            height: 500px; 
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    banner_image_path = "Breast_Cancer_Detection/assets/banner4.jpg"
    st.markdown(f'<img src="data:image/jpg;base64,{base64.b64encode(open(banner_image_path, "rb").read()).decode()}" class="full-width-image">', unsafe_allow_html=True)

    # Load data to display an example
    breast = load_data("Breast_Cancer_Detection/assets/breast_cancer.csv")
    breast.drop('Unnamed: 32', axis=1, inplace=True)
    example = breast.iloc[0] 
    example_values = example[:-1] 
    example_df = pd.DataFrame([example_values], columns=example.index[:-1])
    
    st.markdown("")
    st.markdown("#### Example Feature Values:")
    st.dataframe(example_df, use_container_width=True)

    feature_input = st.text_input(
        label="Enter feature values separated by commas...",
        placeholder="e.g., 15.3, 7.4, 2.1, 3.5",
        key="feature_input"
    )

    if st.button("Predict"):
        feature_values = st.session_state.feature_input
        try:
            feature_values = list(map(float, feature_values.split(',')))
            if len(feature_values) != len(example_df.columns):
                st.error(f"Please enter exactly {len(example_df.columns)} values.")
            else:
                result = get_breast_cancer_prediction(feature_values)
                st.markdown(f"### Prediction: {result}")
        except ValueError:
            st.error("Please enter valid numeric values separated by commas.")

def create_forest_cover_ui():
    """Create the Streamlit UI for forest cover classification."""
    
    st.subheader("Forest Cover Classification")

    # Add custom CSS for controlling image height
    st.markdown(
        """
        <style>
        .full-width-image {
            width: 100%;
            height: 500px;
            object-fit: cover;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    banner_image_path = "Forest_Cover_Classification/assets/banner.jpg"
    st.markdown(f'<img src="data:image/jpg;base64,{base64.b64encode(open(banner_image_path, "rb").read()).decode()}" class="full-width-image">', unsafe_allow_html=True)

    # Load data to display an example
    df = load_data("Forest_Cover_Classification/assets/forest_covertype.csv")
    
    example = df.iloc[0]  
    example_values = example[:-1] 
    example_df = pd.DataFrame([example_values], columns=example.index[:-1])
    
    st.markdown("")
    st.markdown("#### Example Feature Values:")
    st.dataframe(example_df, use_container_width=True)  

    feature_input = st.text_input(
        label="Enter feature values separated by commas...",
        placeholder="e.g., 5.0, 7.4, 2.1, 3.5",
        key="feature_input_forest"
    )

    if st.button("Predict"):
        feature_values = st.session_state.feature_input_forest
        try:
            feature_values = list(map(float, feature_values.split(',')))
            if len(feature_values) != len(example_df.columns):
                st.error(f"Please enter exactly {len(example_df.columns)} values.")
            else:
                cover_name = get_forest_cover_prediction(feature_values)
                st.markdown(f"### Predicted Cover Type: {cover_name[0]}")
                image_url = f"Forest_Cover_Classification/assets/{cover_name[1]}"
                st.image(image_url,width=700)
        except ValueError:
            st.error("Please enter valid numeric values separated by commas.")

def create_crop_recommendation_ui():
    """Create the Streamlit UI for crop recommendation with a banner image below the title."""

    st.subheader("Crop Recommendation")

    # Add custom CSS for controlling image height
    st.markdown(
        """
        <style>
        .full-width-image {
            width: 100%;
            height: 500px; 
            object-fit: cover; 
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    banner_image_path = "Crop_Recommendation/assets/banner2.jpg"
    st.markdown(f'<img src="data:image/jpg;base64,{base64.b64encode(open(banner_image_path, "rb").read()).decode()}" class="full-width-image">', unsafe_allow_html=True)

    # Load data to display an example
    df = load_data("Crop_Recommendation/assets/Crop_recommendation.csv")

    example = df.iloc[0]
    example_values = example[:-1]  
    example_df = pd.DataFrame([example_values], columns=example.index[:-1])
    
    st.markdown("")
    st.markdown("#### Example Feature Values:")
    st.dataframe(example_df, use_container_width=True)


    st.markdown("#### Predict Your Crop:")
    labels = ["Nitrogen", "Phosphorus", "Potassium", "Temperature", "Humidity", "PH", "Rainfall"]

    feature_values = {}

    col1, col2 = st.columns(2)

    for i, label in enumerate(labels[:-1]):  
        if i % 2 == 0:
            with col1:
                feature_values[label] = st.text_input(
                    label=f"{label}:",
                    key=f"{label}_input"
                )
        else:
            with col2:
                feature_values[label] = st.text_input(
                    label=f"{label}:",
                    key=f"{label}_input"
                )

    feature_values[labels[-1]] = st.text_input(
        label=f"{labels[-1]}:",
        key=f"{labels[-1]}_input"
    )
    
    if st.button("Predict Crop"):
        try:
            feature_values_list = []
            for feature in labels:
                value = feature_values[feature].strip()  
                if value:
                    feature_values_list.append(float(value))
                else:
                    st.error(f"Feature '{feature}' is missing a value.")
                    return
            
            if len(feature_values_list) != len(labels):
                st.error(f"Please enter values for all {len(labels)} features.")
            else:
                crop_recommendation = get_crop_recommendation(feature_values_list)
                st.markdown(f"**Recommended Crop:** {crop_recommendation}")
        except ValueError as e:
            st.error(f"Invalid input detected: {e}")


def create_crop_yield_prediction_ui():
    """Create the Streamlit UI for crop yield prediction with an example feature display and separate input boxes for each feature."""

    st.subheader("Crop Yield Prediction")

    # Add custom CSS for controlling image height
    st.markdown(
        """
        <style>
        .full-width-image {
            width: 100%;
            height: 500px;
            object-fit: cover;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    banner_image_path = "Crop_Yield_Prediction/assets/banner.jpg"
    st.markdown(f'<img src="data:image/jpg;base64,{base64.b64encode(open(banner_image_path, "rb").read()).decode()}" class="full-width-image">', unsafe_allow_html=True)

    st.markdown("")
    st.markdown("Example: 1990  1485  121  37.20  Albania  Maize")

    st.markdown("#### Predict Your Crop Yield:")
    labels = ['Year', 'Average Rain Fall (mm/year)', 'Pesticides Tonnes', 'Avg_Temp', 'Area', 'Item']
    feature_values = {}

    col1, col2 = st.columns(2)

    for i, label in enumerate(labels):  
        if i % 2 == 0:
            with col1:
                feature_values[label] = st.text_input(
                    label=f"{label}:",
                    key=f"{label}_input"
                )
        else:
            with col2:
                feature_values[label] = st.text_input(
                    label=f"{label}:",
                    key=f"{label}_input"
                )


    if st.button("Predict"):
        try:
            feature_values_list = []
            for feature in labels:
                value = feature_values[feature].strip()  
                if value:
                    feature_values_list.append(value if feature in ['Area', 'Item'] else float(value))
                else:
                    st.error(f"Feature '{feature}' is missing a value.")
                    return
            
            if len(feature_values_list) != len(labels):
                st.error(f"Please enter values for all {len(labels)} features.")
            else:
                predicted_yield = get_crop_yield_prediction(feature_values_list)
                st.markdown(f"### Predicted Crop Yield: {predicted_yield} hg/ha")
        except ValueError as e:
            st.error(f"Invalid input detected: {e}")

def create_credit_card_fraud_ui():
    """Create the Streamlit UI for credit card fraud detection."""
    
    st.subheader("Credit Card Fraud Detection")

    # Add custom CSS for controlling image height
    st.markdown(
        """
        <style>
        .full-width-image {
            width: 100%;
            height: 500px; 
            object-fit: cover;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    banner_image_path = "Credit_Card_Fraud_Detection/assets/banner.jpg"
    st.markdown(f'<img src="data:image/jpg;base64,{base64.b64encode(open(banner_image_path, "rb").read()).decode()}" class="full-width-image">', unsafe_allow_html=True)

    # Load data to display an example
    df = load_data("Credit_Card_Fraud_Detection/assets/creditcard.csv")
    example = df.iloc[0]  
    example_values = example[:-1]  
    example_df = pd.DataFrame([example_values], columns=example.index[:-1])
    
    st.markdown("")
    st.markdown("#### Example Feature Values:")
    st.dataframe(example_df, use_container_width=True)  

    feature_input = st.text_input(
        label="Enter feature values separated by commas...",
        placeholder="e.g., 0.0, 1.0, 2.3, 3.4, 4.5",
        key="feature_input_fraud"
    )

    if st.button("Predict Fraud"):
        feature_values = st.session_state.feature_input_fraud
        try:
            feature_values = list(map(float, feature_values.split(',')))
            if len(feature_values) != len(example_df.columns):
                st.error(f"Please enter exactly {len(example_df.columns)} values.")
            else:
                result = get_credit_card_fraud_prediction(feature_values)
                st.markdown(f"### Prediction: {result}")
        except ValueError:
            st.error("Please enter valid numeric values separated by commas.")



# Main Page Start Here -->
st.set_page_config(page_title="AI Platform", layout="wide")

def main():
    st.title("AI Platform")
    option = st.sidebar.selectbox(
        "Select a Project", 
        ["Forest Cover Classification", "Breast Cancer Detection", "Crop Recommendation", "Crop Yield Prediction", "Credit Card Fraud Detection"]
    )

    if option == "Forest Cover Classification":
        create_forest_cover_ui()
    elif option == "Breast Cancer Detection":
        create_breast_cancer_ui()
    elif option == "Crop Recommendation":
        create_crop_recommendation_ui()
    elif option == "Crop Yield Prediction":
        create_crop_yield_prediction_ui()
    elif option == "Credit Card Fraud Detection":
        create_credit_card_fraud_ui()

if __name__ == "__main__":
    main()

