import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

def load_processed_data(df):
    """
    Take already wrangled dataframe and prepare it for modeling/visualization
    
    Args:
        df: Pandas DataFrame returned from wrangle function
        
    Returns:
        dict: Dictionary containing training and test splits of data
    """
    # Filter data as in original code
    mask = df['USER_HEIGHT'] < 180
    df = df[mask]
    mask = df['HAND_SPEED'] < 100
    df = df[mask]
    
    # Split data
    X_data = df.drop("SCORE", axis=1)
    target = "SCORE"
    y_data = df[target]
    
    # Create train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.2, random_state=42
    )
    
    # Calculate baseline metrics
    y_mean = y_train.mean()
    y_pred_baseline = [y_mean] * len(y_train)
    baseline_mae = mean_absolute_error(y_train, y_pred_baseline)
    
    return {
        'df': df,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'baseline_mae': baseline_mae,
        'target_mean': y_mean
    }

def calculate_feature_variance(df):
    """
    Calculate variance for each feature in the dataframe
    
    Args:
        df: Pandas DataFrame
        
    Returns:
        pd.Series: Series containing variance of each column, sorted in descending order
    """
    # Remove non-numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Calculate variance
    variances = numeric_df.var()
    
    # Sort by variance in descending order
    sorted_variances = variances.sort_values(ascending=False)
    
    return sorted_variances

def get_top_n_features(df, n=10, by='variance'):
    """
    Get top N features based on specified metric
    
    Args:
        df: Pandas DataFrame
        n: Number of features to return
        by: Metric to use for ranking ('variance' or 'correlation')
        
    Returns:
        list: List of feature names
    """
    if by == 'variance':
        variances = calculate_feature_variance(df)
        return variances.head(n).index.tolist()
    elif by == 'correlation':
        if 'SCORE' in df.columns:
            correlations = df.corr()['SCORE'].abs()
            return correlations.sort_values(ascending=False)[1:n+1].index.tolist()
        else:
            raise ValueError("Cannot calculate correlation with SCORE: SCORE column not found")
    else:
        raise ValueError(f"Unsupported ranking metric: {by}")
