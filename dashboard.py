import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px

# Import custom modules
import wrangle
from data_processing import load_processed_data, calculate_feature_variance, get_top_n_features
from modeling import create_model, train_model, evaluate_model, get_feature_importance
from visualization import (
    plot_score_distribution, 
    plot_feature_vs_score, 
    plot_pairwise_relationships,
    plot_feature_importance,
    plot_variance_ranking,
    create_interactive_correlation_heatmap,
    create_interactive_feature_explorer
)

def main():
    st.set_page_config(page_title="Golf Score Analysis Dashboard", layout="wide")
    
    st.title("Golf Score Analysis Dashboard")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # File upload option
    uploaded_file = st.sidebar.file_uploader("Upload SQLite database file", type=["db"])
    db_path = st.sidebar.text_input("Or enter database path:", "/home/blueaz/Downloads/SensorDownload/May2024/Golf3.db")
    
    # Load data
    if uploaded_file is not None:
        # Save uploaded file temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.getvalue())
            db_path = tmp.name
    
    if not db_path:
        st.warning("Please upload a database file or provide a valid path.")
        return
    
    try:
        with st.spinner("Loading data..."):
            # Load and wrangle data
            df = wrangle.wrangle(db_path)
            data = load_processed_data(df)
            
            # Calculate feature variance
            feature_variances = calculate_feature_variance(df)
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return
    
    # Data overview
    st.header("Data Overview")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Dataset Information")
        st.write(f"Total rows: {df.shape[0]}")
        st.write(f"Total features: {df.shape[1]}")
        st.write(f"Average score: {data['target_mean']:.2f}")
        st.write(f"Baseline MAE: {data['baseline_mae']:.2f}")
    
    with col2:
        st.subheader("Score Distribution")
        fig = plot_score_distribution(df)
        st.pyplot(fig)
    
    # Feature ranking and selection
    st.header("Feature Ranking")
    
    # Feature selection options
    rank_method = st.sidebar.selectbox(
        "Rank features by:",
        ["variance", "correlation", "importance"]
    )
    
    n_features = st.sidebar.slider("Number of top features to display:", 5, 20, 10)
    
    # Train model if needed for importance ranking
    if rank_method == "importance":
        with st.spinner("Training model to calculate feature importance..."):
            model = create_model()
            trained_model = train_model(model, data['X_train'], data['y_train'])
            eval_results = evaluate_model(trained_model, data['X_test'], data['y_test'])
            feature_importance = get_feature_importance(trained_model)
            
            st.success(f"Model trained! Test MAE: {eval_results['mae']:.2f}")
            
            # Display feature importance
            st.subheader("Feature Importance (Absolute Coefficient Values)")
            fig, summary_stats = plot_feature_importance(feature_importance, n=n_features)
            st.pyplot(fig)
            
            # Display summary statistics
            st.subheader("Feature Importance Summary Statistics")
            st.table(pd.DataFrame(summary_stats, index=["Value"]).T)
    
    # Display variance ranking
    if rank_method == "variance":
        st.subheader("Features Ranked by Variance")
        
        # Option to skip top variance features
        skip_top = st.sidebar.slider("Skip top N high variance features:", 0, 5, 0, 
                                    help="Use this to remove outlier features with extremely high variance")
        
        fig, summary_stats = plot_variance_ranking(feature_variances, n=n_features, skip_top=skip_top)
        st.pyplot(fig)
        
        # Display summary statistics
        st.subheader("Variance Summary Statistics")
        st.table(pd.DataFrame(summary_stats, index=["Value"]).T)
        
        # Get top features (considering skipped ones)
        if skip_top > 0:
            top_features = feature_variances.iloc[skip_top:skip_top+n_features].index.tolist()
        else:
            top_features = feature_variances.head(n_features).index.tolist()
    
    # Display correlation ranking
    if rank_method == "correlation":
        st.subheader("Features Ranked by Correlation with Score")
        correlations = df.corr()['SCORE'].abs().sort_values(ascending=False)
        correlations = correlations[correlations.index != 'SCORE'] # Remove SCORE itself
        
        fig, ax = plt.subplots(figsize=(10, 8))
        correlations.head(n_features).plot(kind="barh", ax=ax)
        ax.set_xlabel("Absolute Correlation")
        ax.set_ylabel("Feature")
        ax.set_title(f"Top {n_features} Features by Correlation with Score")
        plt.tight_layout()
        st.pyplot(fig)
        
        top_features = correlations.head(n_features).index.tolist()
    
    # Get top features based on selected method
    if rank_method == "importance":
        top_features = feature_importance.sort_values(key=abs, ascending=False).head(n_features).index.tolist()
    elif rank_method == "variance":
        top_features = feature_variances.head(n_features).index.tolist()
    else:  # correlation
        top_features = df.corr()['SCORE'].abs().sort_values(ascending=False)[1:n_features+1].index.tolist()
    
    # Visual exploration
    st.header("Visual Exploration")
    
    # Feature to explore
    st.subheader("Individual Feature vs. Score")
    selected_feature = st.selectbox("Select a feature to explore:", top_features)
    
    # Create individual feature plot
    fig = plot_feature_vs_score(df, selected_feature)
    st.pyplot(fig)
    
    # Multiple features exploration
    st.subheader("Multi-feature Exploration")
    
    tab1, tab2, tab3 = st.tabs(["Pairplot", "Correlation Matrix", "Interactive Explorer"])
    
    with tab1:
        st.write("Pairwise relationships between top features and Score")
        num_pairs = st.slider("Number of top features for pairplot:", 2, 6, 3)
        features_for_pair = top_features[:num_pairs]
        
        fig = plot_pairwise_relationships(df, features_for_pair, target="SCORE")
        st.pyplot(fig.fig)
    
    with tab2:
        st.write("Correlation matrix of numeric features")
        corr_fig = create_interactive_correlation_heatmap(df)
        st.plotly_chart(corr_fig, use_container_width=True)
    
    with tab3:
        st.write("Interactive feature explorer")
        num_interactive = st.slider("Number of top features for interactive explorer:", 2, 6, 3)
        features_for_interactive = top_features[:num_interactive]
        
        interactive_fig = create_interactive_feature_explorer(df, features_for_interactive, target="SCORE")
        st.plotly_chart(interactive_fig, use_container_width=True)
    
    # Advanced analysis
    st.header("Advanced Analysis")
    
    # Time series view
    st.subheader("Score over Time")
    
    # Get score trends over time
    fig, ax = plt.subplots(figsize=(12, 6))
    df['SCORE'].plot(ax=ax)
    ax.set_xlabel("Date")
    ax.set_ylabel("Score")
    ax.set_title("Score Progression Over Time")
    st.pyplot(fig)
    
    # Custom feature combination analysis
    st.subheader("Custom Feature Combination Analysis")
    
    custom_x = st.selectbox("Select X feature:", df.columns)
    custom_y = st.selectbox("Select Y feature:", [col for col in df.columns if col != custom_x])
    color_by = st.checkbox("Color by Score", value=True)
    
    fig = px.scatter(
        df.reset_index(), 
        x=custom_x, 
        y=custom_y,
        color='SCORE' if color_by else None,
        hover_data=['SCORE'],
        title=f"{custom_y} vs {custom_x}"
    )
    
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
