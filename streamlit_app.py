import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import os
import sys
import tempfile
import io
from pathlib import Path

# Import custom modules
from config import Config
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

# Set page configuration
st.set_page_config(
    page_title="Golf Score Analysis Dashboard",
    page_icon="🏌️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load configuration
config = Config()

# Add custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #3498db;
        text-align: center;
        margin-bottom: 1rem;
    }
    .section-header {
        color: #2ecc71;
        font-size: 1.8rem;
        margin-top: 2rem;
    }
    .stat-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Main title
st.markdown("<h1 class='main-header'>Golf Score Analysis Dashboard</h1>", unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.image("https://www.svgrepo.com/show/133519/golf.svg", width=100)
st.sidebar.header("Dashboard Configuration")

# About section
with st.sidebar.expander("About This Dashboard"):
    st.write("""
    This dashboard analyzes golf swing data to identify relationships between various 
    swing metrics and scores. Upload a database file or use the default path to get started.
    
    Features include:
    - Feature ranking by variance, correlation, and model importance
    - Visual exploration of feature relationships
    - Interactive plots and data exploration
    - Model training and evaluation
    """)

# File upload section
upload_section = st.sidebar.expander("Data Source", expanded=True)
with upload_section:
    uploaded_file = st.file_uploader("Upload SQLite database file", type=["db"])
    use_default_path = st.checkbox("Use default database path", value=True)
    
    if use_default_path:
        db_path = str(config.DB_PATH)
        st.info(f"Using default path: {db_path}")
    else:
        db_path = st.text_input("Enter database path:", "")

# Function to load data
@st.cache_data(ttl=3600)
def cached_load_data(db_path):
    try:
        df = wrangle.wrangle(db_path)
        data = load_processed_data(df)
        feature_variances = calculate_feature_variance(df)
        return df, data, feature_variances
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None

# Main function
def main():
    # Process data source
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as tmp:
            tmp.write(uploaded_file.getvalue())
            temp_db_path = tmp.name
            db_path_to_use = temp_db_path
    elif use_default_path and db_path:
        db_path_to_use = db_path
    else:
        st.warning("Please upload a database file or provide a valid path.")
        return

    # Load data with spinner and caching
    with st.spinner("Loading and processing data..."):
        df, data, feature_variances = cached_load_data(db_path_to_use)
        
    if df is None:
        return

    # Analysis options
    analysis_section = st.sidebar.expander("Analysis Options", expanded=True)
    with analysis_section:
        rank_method = st.selectbox(
            "Rank features by:",
            ["variance", "correlation", "importance"],
            index=["variance", "correlation", "importance"].index(config.DEFAULT_RANK_METHOD)
        )
        
        n_features = st.slider(
            "Number of top features to display:", 
            5, 30, config.DEFAULT_N_FEATURES
        )
        
        if rank_method == "variance":
            skip_top = st.slider(
                "Skip top N high variance features:", 
                0, 5, 0, 
                help="Use this to remove outlier features with extremely high variance"
            )
        
        show_summary = st.checkbox("Show summary statistics", value=True)
        plot_time_series = st.checkbox("Show score progression over time", value=True)

    # Tab views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Data Overview", 
        "🔍 Feature Ranking", 
        "📈 Visual Exploration",
        "🧪 Advanced Analysis"
    ])

    # Tab 1: Data Overview
    with tab1:
        st.markdown("<h2 class='section-header'>Dataset Information</h2>", unsafe_allow_html=True)
        
        # Display basic stats in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("<div class='stat-container'>", unsafe_allow_html=True)
            st.metric("Total Swings", f"{df.shape[0]:,}")
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col2:
            st.markdown("<div class='stat-container'>", unsafe_allow_html=True)
            st.metric("Total Features", f"{df.shape[1]:,}")
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col3:
            st.markdown("<div class='stat-container'>", unsafe_allow_html=True)
            st.metric("Average Score", f"{data['target_mean']:.2f}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        # More data info
        with st.expander("Dataset Structure", expanded=False):
            st.write("DataFrame Info:")
            buffer = io.StringIO()
            df.info(buf=buffer)
            st.text(buffer.getvalue())
            
            st.write("Sample Data:")
            st.dataframe(df.head(), use_container_width=True)
            
            st.write("Data Types:")
            st.dataframe(pd.DataFrame(df.dtypes, columns=["Data Type"]), use_container_width=True)
        
        # Score distribution
        st.markdown("<h2 class='section-header'>Score Distribution</h2>", unsafe_allow_html=True)
        fig = plot_score_distribution(df)
        st.pyplot(fig)
        
        # Time series plot if selected
        if plot_time_series:
            st.markdown("<h2 class='section-header'>Score Progression Over Time</h2>", unsafe_allow_html=True)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            df['SCORE'].plot(ax=ax)
            ax.set_xlabel("Date")
            ax.set_ylabel("Score")
            ax.set_title("Score Progression Over Time")
            st.pyplot(fig)
            
            # Show rolling average
            window_size = st.slider("Rolling average window size:", 2, 20, 5)
            if window_size > 1:
                fig, ax = plt.subplots(figsize=(12, 6))
                df['SCORE'].plot(ax=ax, alpha=0.5, label='Raw scores')
                df['SCORE'].rolling(window=window_size).mean().plot(ax=ax, linewidth=2, label=f'{window_size}-swing rolling average')
                ax.set_xlabel("Date")
                ax.set_ylabel("Score")
                ax.set_title(f"Score Progression with {window_size}-swing Rolling Average")
                ax.legend()
                st.pyplot(fig)

    # Tab 2: Feature Ranking
    with tab2:
        st.markdown("<h2 class='section-header'>Feature Ranking</h2>", unsafe_allow_html=True)
        
        # Train model if needed for importance ranking
        if rank_method == "importance":
            with st.spinner("Training model to calculate feature importance..."):
                model = create_model()
                trained_model = train_model(model, data['X_train'], data['y_train'])
                eval_results = evaluate_model(trained_model, data['X_test'], data['y_test'])
                feature_importance = get_feature_importance(trained_model)
                
                # Display model metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Test MAE", f"{eval_results['mae']:.2f}")
                with col2:
                    st.metric("Baseline MAE", f"{data['baseline_mae']:.2f}")
                with col3:
                    improvement = ((data['baseline_mae'] - eval_results['mae']) / data['baseline_mae'] * 100)
                    st.metric("Improvement", f"{improvement:.1f}%")
                
                # Display feature importance
                st.markdown("<h3>Feature Importance (Absolute Coefficient Values)</h3>", unsafe_allow_html=True)
                fig, summary_stats = plot_feature_importance(feature_importance, n=n_features)
                st.pyplot(fig)
                
                # Get top features
                top_features = feature_importance.sort_values(key=abs, ascending=False).head(n_features).index.tolist()
                
                # Display summary statistics if selected
                if show_summary:
                    st.subheader("Feature Importance Summary Statistics")
                    st.table(pd.DataFrame(summary_stats, index=["Value"]).T)
        
        # Display variance ranking
        elif rank_method == "variance":
            st.markdown("<h3>Features Ranked by Variance</h3>", unsafe_allow_html=True)
            
            fig, summary_stats = plot_variance_ranking(feature_variances, n=n_features, skip_top=skip_top)
            st.pyplot(fig)
            
            # Get top features (considering skipped ones)
            if skip_top > 0:
                top_features = feature_variances.iloc[skip_top:skip_top+n_features].index.tolist()
            else:
                top_features = feature_variances.head(n_features).index.tolist()
            
            # Display summary statistics if selected
            if show_summary:
                st.subheader("Variance Summary Statistics")
                st.table(pd.DataFrame(summary_stats, index=["Value"]).T)
        
        # Display correlation ranking
        elif rank_method == "correlation":
            st.markdown("<h3>Features Ranked by Correlation with Score</h3>", unsafe_allow_html=True)
            correlations = df.corr()['SCORE'].abs().sort_values(ascending=False)
            correlations = correlations[correlations.index != 'SCORE']  # Remove SCORE itself
            
            fig, ax = plt.subplots(figsize=(10, 8))
            correlations.head(n_features).plot(kind="barh", ax=ax)
            ax.set_xlabel("Absolute Correlation")
            ax.set_ylabel("Feature")
            ax.set_title(f"Top {n_features} Features by Correlation with Score")
            plt.tight_layout()
            st.pyplot(fig)
            
            # Get top features
            top_features = correlations.head(n_features).index.tolist()
            
            # Display summary statistics if selected
            if show_summary:
                st.subheader("Correlation Summary Statistics")
                summary_stats = {
                    "Min Correlation (abs)": correlations.min(),
                    "Max Correlation (abs)": correlations.max(),
                    "Mean Correlation (abs)": correlations.mean(),
                    "Median Correlation (abs)": correlations.median(),
                    "Std Dev": correlations.std(),
                    "Total Features": len(correlations)
                }
                st.table(pd.DataFrame(summary_stats, index=["Value"]).T)
        
        # Display top features in table form
        st.subheader(f"Top {len(top_features)} Features by {rank_method.capitalize()}")
        feature_df = pd.DataFrame({"Feature": top_features})
        st.dataframe(feature_df, use_container_width=True)

    # Tab 3: Visual Exploration
    with tab3:
        st.markdown("<h2 class='section-header'>Visual Exploration</h2>", unsafe_allow_html=True)
        
        # Feature to explore
        st.subheader("Individual Feature vs. Score")
        selected_feature = st.selectbox("Select a feature to explore:", top_features)
        
        # Create individual feature plot
        fig = plot_feature_vs_score(df, selected_feature)
        st.pyplot(fig)
        
        # Show statistics for selected feature
        if show_summary:
            st.subheader(f"Statistics for {selected_feature}")
            
            feature_stats = {
                "Min": df[selected_feature].min(),
                "Max": df[selected_feature].max(),
                "Mean": df[selected_feature].mean(),
                "Median": df[selected_feature].median(),
                "Std Dev": df[selected_feature].std(),
                "Correlation with Score": df[[selected_feature, "SCORE"]].corr().iloc[0, 1]
            }
            
            st.table(pd.DataFrame(feature_stats, index=["Value"]).T)
        
        # Multiple features exploration
        st.markdown("<h2 class='section-header'>Multi-feature Exploration</h2>", unsafe_allow_html=True)
        
        subtab1, subtab2, subtab3 = st.tabs(["Pairplot", "Correlation Matrix", "Interactive Explorer"])
        
        with subtab1:
            st.write("Pairwise relationships between top features and Score")
            num_pairs = st.slider("Number of top features for pairplot:", 2, 
                                 min(config.MAX_FEATURES_PAIRPLOT, len(top_features)), 3)
            features_for_pair = top_features[:num_pairs]
            
            with st.spinner("Generating pairplot..."):
                fig = plot_pairwise_relationships(df, features_for_pair, target="SCORE")
                st.pyplot(fig.fig)
        
        with subtab2:
            st.write("Correlation matrix of numeric features")
            with st.spinner("Generating correlation heatmap..."):
                corr_fig = create_interactive_correlation_heatmap(df)
                st.plotly_chart(corr_fig, use_container_width=True)
        
        with subtab3:
            st.write("Interactive feature explorer")
            num_interactive = st.slider("Number of top features for interactive explorer:", 2, 
                                      min(config.MAX_FEATURES_PAIRPLOT, len(top_features)), 3)
            features_for_interactive = top_features[:num_interactive]
            
            with st.spinner("Generating interactive explorer..."):
                interactive_fig = create_interactive_feature_explorer(df, features_for_interactive, target="SCORE")
                st.plotly_chart(interactive_fig, use_container_width=True)

    # Tab 4: Advanced Analysis
    with tab4:
        st.markdown("<h2 class='section-header'>Advanced Analysis</h2>", unsafe_allow_html=True)
        
        # Custom feature combination analysis
        st.subheader("Custom Feature Combination Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            custom_x = st.selectbox("Select X feature:", df.columns)
        with col2:
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
        
        # Grouped analysis
        st.subheader("Grouped Analysis")
        
        # Let user select a categorical feature to group by
        categorical_cols = [col for col in df.columns if df[col].nunique() < 10]
        if categorical_cols:
            group_by = st.selectbox("Group by feature:", categorical_cols)
            
            # Create grouped plot
            fig, ax = plt.subplots(figsize=(12, 6))
            df.groupby(group_by)['SCORE'].mean().sort_values().plot(kind='bar', ax=ax)
            ax.set_xlabel(group_by)
            ax.set_ylabel("Average Score")
            ax.set_title(f"Average Score by {group_by}")
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("No suitable categorical columns found for grouping")
        
        # Download Data
        st.subheader("Download Analysis Data")
        
        # Create analysis results
        if st.button("Generate Analysis Report"):
            # Create a DataFrame with statistics
            stats_df = pd.DataFrame({
                "Feature": df.columns,
                "Min": df.min(),
                "Max": df.max(),
                "Mean": df.mean(),
                "Median": df.median(),
                "Std Dev": df.std(),
                "Variance": df.var(),
                "Correlation with Score": [df[[col, "SCORE"]].corr().iloc[0, 1] if col != "SCORE" else 1.0 for col in df.columns]
            })
            
            # Convert to CSV
            csv = stats_df.to_csv(index=False)
            
            # Provide download link
            st.download_button(
                label="Download Analysis CSV",
                data=csv,
                file_name="golf_score_analysis.csv",
                mime="text/csv",
            )

    # Footer
    st.markdown("---")
    st.markdown(
        "Golf Score Analysis Dashboard | Created with Streamlit | Data refreshed: " + 
        pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    )

if __name__ == "__main__":
    main()
