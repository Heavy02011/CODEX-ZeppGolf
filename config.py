from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    """Configuration settings for the Golf Swing Analysis Dashboard"""
    
    # Database paths
    DB_PATH: Path = Path(__file__).parent / 'Golf3.db'  # Path to database in current directory
    
    # Time and timezone settings
    TIMEZONE: str = 'America/Phoenix'
    
    # Data filtering thresholds
    MAX_USER_HEIGHT: float = 180.0  # Height threshold for filtering users
    MAX_HAND_SPEED: float = 100.0   # Maximum hand speed threshold
    
    # Modeling settings
    TEST_SIZE: float = 0.2          # Percentage of data to use for testing
    RANDOM_SEED: int = 42           # Random seed for reproducibility
    
    # Visualization settings
    DEFAULT_N_FEATURES: int = 10    # Default number of features to display
    MAX_FEATURES_PAIRPLOT: int = 6  # Maximum number of features for pairplot
    
    # Dashboard settings
    DEFAULT_RANK_METHOD: str = "variance"  # Default method to rank features
    SKIP_TOP_VARIANCES: int = 0     # Default number of top variance features to skip
    SHOW_SUMMARY_STATS: bool = True # Whether to show summary statistics by default
    
    # Plot colors
    PLOT_COLORS: dict = None
    
    def __post_init__(self):
        self.PLOT_COLORS = {
            'score': '#2ecc71',
            'hand_speed': '#e74c3c',
            'impact_speed': '#3498db',
            'club_face': '#e67e22',
            'swing_tempo': '#9b59b6',
            'club_plane': '#f1c40f',
            'twist': '#1abc9c',
            'posture': '#34495e',
            'transition': '#7f8c8d',
            'upswing': '#d35400'
        }
