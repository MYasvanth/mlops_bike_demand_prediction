#!/usr/bin/env python3
"""
Setup conda environment for MLOps Bike Demand project.
"""

import subprocess
import sys
import os

def run_command(cmd):
    """Run command and return success status."""
    try:
        subprocess.run(cmd, shell=True, check=True)
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    print("🐍 Setting up Conda Environment for MLOps Bike Demand")
    print("=" * 55)
    
    # Create conda environment
    print("Creating conda environment...")
    if run_command("conda env create -f environment.yml"):
        print("✅ Conda environment created successfully")
    else:
        print("❌ Failed to create conda environment")
        return False
    
    print("\n🎉 Setup completed!")
    print("\nTo activate the environment:")
    print("conda activate mlops_bike_demand")
    print("\nTo run the project:")
    print("python run_pipeline.py")
    
    return True

if __name__ == "__main__":
    main()