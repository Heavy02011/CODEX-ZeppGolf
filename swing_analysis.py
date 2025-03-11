import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import logging
import os
import time
from pathlib import Path
import sys

# Add the current directory to the path so we can import the modules
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

# Import our modules
from data_parser import GolfDataParser
from visualization_module import GolfSwingVisualizer
from file_manager import GolfSwingFileManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("golf_streamlit.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("golf_streamlit")

# Initialize session state for persistence between reruns
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.file_manager = None
    st.session_state.visualizer = None
    st.session_state.data_parser = None
    st.session_state.current_swing_id = None
    st.session_state.current_data = None
    st.session_state.view_type = 'hand_path'
    st.session_state.frame = 0
    st.session_state.h_angle = 45
    st.session_state.v_angle = 30
    st.session_state.zoom = 1.5
    st.session_state.play_animation = False
    st.session_state.last_animation_time = time.time()

def initialize_app():
    """Initialize the app components"""
    if not st.session_state.initialized:
        logger.info("Initializing app components")
        
        # Initialize file manager
        st.session_state.file_manager = GolfSwingFileManager()
        
        # Initialize visualizer
        st.session_state.visualizer = GolfSwingVisualizer()
        
        # Initialize data parser
        st.session_state.data_parser = GolfDataParser()
        
        st.session_state.initialized = True
        logger.info("App components initialized")

def load_swing_data(swing_id):
    """Load swing data and update the session state"""
    logger.info(f"Loading swing data for ID: {swing_id}")
    
    # Update current swing ID
    st.session_state.current_swing_id = swing_id
    
    # Load swing data
    swing_data = st.session_state.file_manager.load_swing_file(swing_id)
    
    # If no data, try to generate sample data
    if swing_data is None:
        logger.warning(f"No data found for swing ID {swing_id}, using sample data")
        swing_data = st.session_state.data_parser.get_sample_data()
    
    # Update session state
    st.session_state.current_data = swing_data
    
    # Update visualizer
    st.session_state.visualizer.set_data(swing_data)
    st.session_state.visualizer.set_view_type(st.session_state.view_type)
    st.session_state.visualizer.set_frame(st.session_state.frame)
    
    return swing_data

def main():
    """Main function to run the Streamlit app"""
    st.set_page_config(
        page_title="Golf Swing Analyzer",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize app components
    initialize_app()
    
    # Page title
    st.title("Golf Swing Analyzer")
    
    # Sidebar
    with st.sidebar:
        st.header("Controls")
        
        # Swing selection
        st.subheader("Select Swing")
        swing_ids = st.session_state.file_manager.get_available_swing_ids()
        
        if not swing_ids:
            st.warning("No swing files found. Using sample data.")
            swing_ids = [0]
            st.session_state.file_manager.create_mock_swing_file()
        
        swing_id = st.selectbox(
            "Swing ID",
            options=swing_ids,
            index=0 if st.session_state.current_swing_id is None else swing_ids.index(st.session_state.current_swing_id),
            key="swing_selector"
        )
        
        # If swing ID has changed, load the new data
        if st.session_state.current_swing_id != swing_id:
            load_swing_data(swing_id)
        
        # View type
        st.subheader("View Settings")
        view_type = st.radio(
            "View Type",
            options=["club_path", "hand_path", "plane_view"],
            index=["club_path", "hand_path", "plane_view"].index(st.session_state.view_type),
            format_func=lambda x: x.replace('_', ' ').title(),
            key="view_type_selector"
        )
        
        # If view type has changed, update the visualizer
        if st.session_state.view_type != view_type:
            st.session_state.view_type = view_type
            if st.session_state.visualizer:
                st.session_state.visual
