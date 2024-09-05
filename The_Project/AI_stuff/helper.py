import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Any
import streamlit as st 

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the dataset from the given file path.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset as a DataFrame.
    """
    return pd.read_csv(filepath)


def preprocess_data(df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Preprocess the data by separating features and the target variable.

    Args:
        df (pd.DataFrame): The dataset to be processed.
        target_column (str): The name of the target column.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features (X) and target variable (y).
    """
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    return X, y


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the data into training and test sets.

    Args:
        X (pd.DataFrame): Feature set.
        y (pd.Series): Target variable.
        test_size (float, optional): Proportion of the data for testing. Defaults to 0.2.
        random_state (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: Training and test sets for features and target variable.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_model(X_train: pd.DataFrame, y_train: pd.Series, model_class: Any, **model_kwargs) -> Any:
    """
    Train a machine learning model.
    
    Parameters:
    - X_train: Training feature data
    - y_train: Training target data
    - model_class: A machine learning model class (e.g., KNeighborsClassifier)
    - model_kwargs: Additional arguments to pass to the model constructor
    
    Returns:
    - Trained model instance
    """
    model = model_class(**model_kwargs)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """
    Evaluate the model and return various performance metrics.
    
    Parameters:
    - model: Trained machine learning model
    - X_test: Test feature data
    - y_test: Test target data
    
    Returns:
    - Dictionary of evaluation metrics (accuracy, precision, recall, f1 score)
    """
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='weighted'),
        "recall": recall_score(y_test, y_pred, average='weighted'),
        "f1_score": f1_score(y_test, y_pred, average='weighted')
    }
    return metrics

def plot_confusion_matrix(y_true: pd.Series, y_pred: pd.Series, labels: list) -> None:
    """
    Plot a confusion matrix using Seaborn and display it in Streamlit.

    Args:
        y_true (pd.Series): True labels.
        y_pred (pd.Series): Predicted labels.
        labels (list): List of label names for the confusion matrix.

    Returns:
        None: Displays the confusion matrix in Streamlit.
    """
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot(plt)
    plt.close()

