import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import Config
import wrangle


def analyze_database(db_path=Config.DB_PATH, output_dir="analysis_output"):
    """Load the entire database, generate a correlation heatmap and save it."""
    os.makedirs(output_dir, exist_ok=True)

    df = wrangle.wrangle(db_path)
    corr = df.corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap")
    fig.tight_layout()
    output_path = os.path.join(output_dir, "correlation_heatmap.png")
    fig.savefig(output_path)
    plt.close(fig)

    print(f"Heatmap saved to {output_path}")


if __name__ == "__main__":
    analyze_database()
