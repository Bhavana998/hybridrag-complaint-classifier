#!/usr/bin/env python
import sys
import subprocess
from pathlib import Path

def main():
    """Main entry point for HybridRAG application"""
    print("🚀 Starting HybridRAG Complaint Classification System...")
    print("=" * 50)
    
    # Check if streamlit is installed
    try:
        import streamlit
    except ImportError:
        print("📦 Installing required packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements/base.txt"])
    
    # Run streamlit app
    app_path = Path(__file__).parent / "app" / "main.py"
    subprocess.run(["streamlit", "run", str(app_path), "--server.port=8501"])

if __name__ == "__main__":
    main()