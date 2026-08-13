import sys
import os

print("Starting Flask Web Dashboard on port 5001...")
from main import app, run_flask

if __name__ == "__main__":
    run_flask()
