import os
import sqlite3

def get_project_root():
    """Returns the absolute path to the project root directory."""
    # This file is in src/, so we go up one level
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(current_dir)

def get_db_path():
    """Returns the path to the sqlite database relative to the project root."""
    return os.path.join(get_project_root(), "data", "risklens.db")

def get_csv_path():
    """Returns the path to the Kaggle credit card CSV file."""
    return os.path.join(get_project_root(), "data", "creditcard.csv")

def get_connection(db_path=None):
    """
    Creates and returns a sqlite3 connection to the database.
    Ensures the target directory exists.
    """
    if db_path is None:
        db_path = get_db_path()
        
    # Ensure data directory exists before connecting
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    return sqlite3.connect(db_path)

# Consistent color palette for data visualizations in dashboards/notebooks
COLOR_PALETTE = {
    "fraud": "#E74C3C",        # Strong Red
    "legit": "#2ECC71",        # Emerald Green
    "warning": "#F1C40F",      # Yellow
    "neutral": "#3498DB",      # Blue
    "background": "#2C3E50"    # Dark Blue/Grey
}
