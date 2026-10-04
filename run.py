"""
run.py
------
Simple entry point for the Market Risk Engine.

Usage:
    python run.py

This starts the FastAPI server which:
1. Runs the full risk engine pipeline (data loading, VaR, ES, backtesting).
2. Serves the REST API at http://localhost:8000/api/
3. Serves the interactive dashboard at http://localhost:8000/

Alternatively:
    uvicorn backend.app:app --reload --port 8000
"""

import sys
import subprocess
from pathlib import Path

def main():
    """Start the uvicorn server for the risk engine."""
    print("=" * 60)
    print("Multi-Asset Market Risk & Tail-Loss Analytics Engine")
    print("=" * 60)
    print()
    print("Starting risk engine and API server...")
    print("Dashboard will be available at: http://localhost:8000")
    print("API documentation at:           http://localhost:8000/docs")
    print()
    print("Press CTRL+C to stop.")
    print()

    # Change to project root directory
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))

    try:
        import uvicorn
        uvicorn.run(
            "backend.app:app",
            host="0.0.0.0",
            port=8000,
            reload=False,  # Set True for development
            log_level="info",
        )
    except ImportError:
        print("uvicorn not found. Run: pip install -r requirements.txt")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    main()
