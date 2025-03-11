from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from category_encoders import OneHotEncoder
from sklearn.metrics import mean_absolute_error
import pandas as pd

def create_model():
    """
    Create a pipeline for modeling
    
    Returns:
        sklearn.pipeline.Pipeline: Model pipeline
    """
    model = make_pipeline(
        OneHotEncoder(use_cat_names=True),
        SimpleImputer(),
        Ridge()   
    )
    return model

def train_model(model, X_train, y_train):
    """
    Train the model
    
    Args:
        model: sklearn.pipeline.Pipeline model
        X_train: Features DataFrame
        y_train: Target Series
        
    Returns:
        model: Trained model
    """
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    
    return {
        'mae': mae,
        'predictions': y_pred
    }

def get_feature_importance(model):
    """
    Extract feature importance from Ridge model
    
    Args:
        model: Trained pipeline with Ridge regression
        
    Returns:
        pd.Series: Series containing feature importance
    """
    coefficients = model.named_steps["ridge"].coef_
    features = model.named_steps["onehotencoder"].get_feature_names()
    feat_imp = pd.Series(coefficients, index=features)
    
    return feat_imp.sort_values(key=abs, ascending=False)
