import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import warnings

# Add current directory to path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import custom modules
import wrangle
from data_processing import load_processed_data, calculate_feature_variance, get_top_n_features
from modeling import create_model, train_model, evaluate_model, get_feature_importance
from visualization import (
    plot_score_distribution, 
    plot_feature_vs_score, 
    plot_pairwise_relationships,
    plot_feature_importance,
    plot_variance_ranking
)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Golf Score Analysis")
    parser.add_argument("--db-path", type=str, required=True, help="Path to SQLite database")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory to save outputs")
    parser.add_argument("--n-features", type=int, default=10, help="Number of top features to display")
    parser.add_argument("--rank-by", type=str, default="variance", 
                        choices=["variance", "correlation", "importance"],
                        help="Method to rank features")
    parser.add_argument("--run-model", action="store_true", help="Train and evaluate model")
    
    return parser.parse_args()

def main():
    """Main function"""
    
    # Parse arguments
    args = parse_args()
    
    # Suppress warnings
    warnings.simplefilter("ignore", UserWarning)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and process data
    print("Loading and processing data...")
    df = wrangle.wrangle(args.db_path)
    data = load_processed_data(df)
    
    # Print basic stats
    print(f"Dataset shape: {df.shape}")
    print(f"Average score: {data['target_mean']:.2f}")
    print(f"Baseline MAE: {data['baseline_mae']:.2f}")
    
    # Plot score distribution
    print("Plotting score distribution...")
    fig = plot_score_distribution(df)
    fig.savefig(os.path.join(args.output_dir, "score_distribution.png"))
    plt.close(fig)
    
    # Feature ranking
    if args.rank_by == "variance":
        print("Ranking features by variance...")
        feature_variances = calculate_feature_variance(df)
        top_features = feature_variances.head(args.n_features).index.tolist()
        
        # Plot variance ranking with option to skip top features
        skip_top = 0  # Can be parameterized via args if needed
        fig, summary_stats = plot_variance_ranking(feature_variances, n=args.n_features, skip_top=skip_top)
        fig.savefig(os.path.join(args.output_dir, "variance_ranking.png"))
        plt.close(fig)
        
        # Save summary statistics
        pd.DataFrame(summary_stats, index=["Value"]).T.to_csv(
            os.path.join(args.output_dir, "variance_summary.csv")
        )
        
    elif args.rank_by == "correlation":
        print("Ranking features by correlation with score...")
        correlations = df.corr()['SCORE'].abs().sort_values(ascending=False)
        correlations = correlations[correlations.index != 'SCORE']  # Remove SCORE itself
        top_features = correlations.head(args.n_features).index.tolist()
        
        # Plot correlation ranking
        fig, ax = plt.subplots(figsize=(10, 8))
        correlations.head(args.n_features).plot(kind="barh", ax=ax)
        ax.set_xlabel("Absolute Correlation")
        ax.set_ylabel("Feature")
        ax.set_title(f"Top {args.n_features} Features by Correlation with Score")
        plt.tight_layout()
        fig.savefig(os.path.join(args.output_dir, "correlation_ranking.png"))
        plt.close(fig)
        
    elif args.rank_by == "importance":
        print("Ranking features by model importance...")
        # Train model
        model = create_model()
        trained_model = train_model(model, data['X_train'], data['y_train'])
        
        # Get feature importance
        feature_importance = get_feature_importance(trained_model)
        top_features = feature_importance.sort_values(key=abs, ascending=False).head(args.n_features).index.tolist()
        
        # Plot feature importance
        fig, summary_stats = plot_feature_importance(feature_importance, n=args.n_features)
        fig.savefig(os.path.join(args.output_dir, "feature_importance.png"))
        plt.close(fig)
        
        # Save summary statistics
        pd.DataFrame(summary_stats, index=["Value"]).T.to_csv(
            os.path.join(args.output_dir, "importance_summary.csv")
        )
    
    # Print top features
    print(f"\nTop {args.n_features} features by {args.rank_by}:")
    for i, feature in enumerate(top_features, 1):
        print(f"{i}. {feature}")
    
    # Individual feature plots
    print("\nCreating individual feature plots...")
    for feature in top_features[:5]:  # Limit to first 5 to avoid too many plots
        fig = plot_feature_vs_score(df, feature)
        fig.savefig(os.path.join(args.output_dir, f"score_vs_{feature}.png"))
        plt.close(fig)
    
    # Pairwise plot of top features
    print("Creating pairwise plot...")
    num_pairs = min(5, len(top_features))  # Limit to 5 features maximum
    features_for_pair = top_features[:num_pairs]
    pair_grid = plot_pairwise_relationships(df, features_for_pair, target="SCORE")
    pair_grid.fig.savefig(os.path.join(args.output_dir, "pairwise_relationships.png"))
    plt.close(pair_grid.fig)
    
    # Train and evaluate model if requested
    if args.run_model:
        print("\nTraining and evaluating model...")
        model = create_model()
        trained_model = train_model(model, data['X_train'], data['y_train'])
        eval_results = evaluate_model(trained_model, data['X_test'], data['y_test'])
        
        print(f"Model test MAE: {eval_results['mae']:.2f}")
        print(f"Improvement over baseline: {((data['baseline_mae'] - eval_results['mae']) / data['baseline_mae'] * 100):.2f}%")
    
    print(f"\nAnalysis complete! Results saved to {args.output_dir}/")

if __name__ == "__main__":
    main()
