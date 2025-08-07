#!/usr/bin/env python3
"""
Launch script for RAG Chatbot Frontend

This script handles the startup of the Streamlit application with proper configuration.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Main launch function"""
    
    # Get the directory of this script
    frontend_dir = Path(__file__).parent
    
    # Change to frontend directory
    os.chdir(frontend_dir)
    
    # Set environment variables
    os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
    os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
    
    # Check if dependencies are installed
    try:
        import streamlit
        import requests
        import pandas
        import plotly
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("📦 Installing frontend dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Launch Streamlit app
    print("🚀 Starting RAG Chatbot Frontend...")
    print("🌐 Open your browser and navigate to: http://localhost:8501")
    print("📱 The interface will open automatically in your default browser")
    print("🔄 Press Ctrl+C to stop the server")
    
    try:
        # Run Streamlit with custom config
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.address", "0.0.0.0",
            "--server.port", "8501",
            "--browser.gatherUsageStats", "false",
            "--theme.primaryColor", "#667eea",
            "--theme.backgroundColor", "#ffffff",
            "--theme.secondaryBackgroundColor", "#f0f2f6",
            "--theme.textColor", "#262730"
        ])
    except KeyboardInterrupt:
        print("\n👋 RAG Chatbot Frontend stopped.")

if __name__ == "__main__":
    main()
