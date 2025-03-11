import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_score_distribution(df):
    """
    Plot distribution of SCORE variable
    
    Args:
        df: DataFrame containing SCORE column
        
    Returns:
        fig: matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df["SCORE"], kde=True, ax=ax)
    ax.set_xlabel("Score")
    ax.set_ylabel("Count")
    ax.set_title("Score Distribution")
    return fig

def plot_feature_vs_score(df, feature):
    """
    Plot relationship between a feature and SCORE
    
    Args:
        df: DataFrame containing data
        feature: Feature to plot against SCORE
        
    Returns:
        fig: matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if df[feature].dtype in ['int64', 'float64']:
        # Numerical feature
        sns.scatterplot(x=df[feature], y=df["SCORE"], ax=ax)
        # Add regression line
        sns.regplot(x=df[feature], y=df["SCORE"], scatter=False, ax=ax)
    else:
        # Categorical feature
        sns.boxplot(x=df[feature], y=df["SCORE"], ax=ax)
    
    ax.set_xlabel(feature)
    ax.set_ylabel("SCORE")
    ax.set_title(f"Score vs {feature}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def plot_pairwise_relationships(df, features, target="SCORE"):
    """
    Create a pairwise plot for multiple features
    
    Args:
        df: DataFrame containing data
        features: List of features to include
        target: Target variable name
        
    Returns:
        grid: seaborn PairGrid
    """
    # Include target in the feature list if not already there
    if target not in features:
        plot_features = features + [target]
    else:
        plot_features = features
        
    # Create pairwise plot
    grid = sns.pairplot(df[plot_features], 
                        hue=target if len(df[target].unique()) <= 10 else None,
                        diag_kind="kde",
                        height=2.5)
    grid.fig.suptitle("Pairwise Relationships", y=1.02, fontsize=16)
    plt.tight_layout()
    return grid

def plot_feature_importance(feature_importance, n=20):
    """
    Plot feature importance
    
    Args:
        feature_importance: Series with feature names as index and importance as values
        n: Number of top features to show
        
    Returns:
        fig: matplotlib figure, summary_stats dictionary
    """
    top_n = feature_importance.sort_values(key=abs).tail(n)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    top_n.plot(kind="barh", ax=ax)
    ax.set_xlabel("Coefficient Value")
    ax.set_ylabel("Feature")
    ax.set_title(f"Top {n} Feature Importance")
    plt.tight_layout()
    
    # Create summary statistics
    abs_importance = feature_importance.abs()
    summary_stats = {
        "Min Importance (abs)": abs_importance.min(),
        "Max Importance (abs)": abs_importance.max(),
        "Mean Importance (abs)": abs_importance.mean(),
        "Median Importance (abs)": abs_importance.median(),
        "Std Dev (abs)": abs_importance.std(),
        "Top Positive Feature": feature_importance.max(),
        "Top Negative Feature": feature_importance.min(),
        "Total Features": len(feature_importance)
    }
    
    return fig, summary_stats

def plot_variance_ranking(feature_variances, n=20, skip_top=0):
    """
    Plot features by variance ranking
    
    Args:
        feature_variances: Series with feature names as index and variances as values
        n: Number of top features to show
        skip_top: Number of top features to skip (useful for very high variance features)
        
    Returns:
        fig: matplotlib figure, summary table text
    """
    # Skip top N features if requested
    if skip_top > 0:
        display_variances = feature_variances.iloc[skip_top:skip_top+n]
        title_suffix = f" (skipping top {skip_top})"
    else:
        display_variances = feature_variances.head(n)
        title_suffix = ""
    
    fig, ax = plt.subplots(figsize=(10, 8))
    display_variances.plot(kind="barh", ax=ax)
    ax.set_xlabel("Variance")
    ax.set_ylabel("Feature")
    ax.set_title(f"Top {n} Features by Variance{title_suffix}")
    plt.tight_layout()
    
    # Create summary statistics
    summary_stats = {
        "Min Variance": feature_variances.min(),
        "Max Variance": feature_variances.max(),
        "Mean Variance": feature_variances.mean(),
        "Median Variance": feature_variances.median(),
        "Std Dev": feature_variances.std()
    }
    
    # Add information about skipped features if applicable
    if skip_top > 0:
        skipped_features = feature_variances.head(skip_top)
        for i, (feature, var) in enumerate(skipped_features.items()):
            summary_stats[f"Skipped #{i+1}: {feature}"] = var
    
    return fig, summary_stats

def create_interactive_correlation_heatmap(df):
    """
    Create interactive correlation heatmap using Plotly
    
    Args:
        df: DataFrame containing data
        
    Returns:
        fig: plotly figure
    """
    # Calculate correlation matrix
    corr = df.select_dtypes(include=['number']).corr()
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Set up figure
    fig = px.imshow(
        corr,
        color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1,
        labels=dict(color="Correlation"),
        title="Feature Correlation Matrix"
    )
    
    # Hide upper triangle
    for i in range(len(corr)):
        for j in range(len(corr)):
            if mask[i, j]:
                fig.data[0].z[i][j] = None
    
    # Update layout
    fig.update_layout(
        width=900,
        height=900,
        autosize=False
    )
    
    return fig

def create_interactive_feature_explorer(df, features, target="SCORE"):
    """
    Create interactive scatter plot matrix using Plotly
    
    Args:
        df: DataFrame containing data
        features: List of features to include
        target: Target variable name
        
    Returns:
        fig: plotly figure
    """
    # Include target in the feature list if not already there
    if target not in features:
        plot_features = features + [target]
    else:
        plot_features = features
    
    # Create scatter plot matrix
    fig = px.scatter_matrix(
        df[plot_features],
        dimensions=plot_features,
        color=target,
        title="Interactive Feature Explorer",
        opacity=0.5
    )
    
    # Update layout
    fig.update_layout(
        width=900,
        height=900,
        autosize=True
    )
    
    # Update traces
    fig.update_traces(diagonal_visible=False)
    
    return fig
