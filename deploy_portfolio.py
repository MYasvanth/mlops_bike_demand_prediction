#!/usr/bin/env python3
"""
Portfolio deployment script for MLOps Bike Demand project
"""
import os
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run shell command with error handling"""
    print(f"🔄 {description}...")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Error: {result.stderr}")
        return False
    print(f"✅ {description} completed")
    return True

def deploy_to_github():
    """Deploy project to GitHub"""
    commands = [
        ("git add .", "Adding files to git"),
        ("git commit -m 'Portfolio: MLOps Bike Demand Prediction - Production Ready'", "Committing changes"),
        ("git push origin main", "Pushing to GitHub")
    ]
    
    for cmd, desc in commands:
        if not run_command(cmd, desc):
            return False
    return True

def create_demo_data():
    """Create sample data for demo"""
    demo_script = """
import pandas as pd
import numpy as np

# Create sample data for demo
np.random.seed(42)
demo_data = {
    'temp': [0.3, 0.5, 0.7],
    'atemp': [0.35, 0.55, 0.75], 
    'hum': [0.6, 0.4, 0.8],
    'windspeed': [0.2, 0.3, 0.1],
    'season': [1, 2, 3],
    'yr': [1, 1, 1],
    'mnth': [6, 7, 8],
    'holiday': [0, 0, 1],
    'weekday': [1, 2, 3],
    'workingday': [1, 1, 0],
    'weathersit': [1, 2, 1]
}

df = pd.DataFrame(demo_data)
df.to_csv('demo_input.csv', index=False)
print("Demo data created: demo_input.csv")
"""
    
    with open('create_demo.py', 'w') as f:
        f.write(demo_script)
    
    run_command("python create_demo.py", "Creating demo data")

def main():
    """Main deployment function"""
    print("🚀 Deploying MLOps Bike Demand to Portfolio...")
    
    # Check if git is initialized
    if not Path('.git').exists():
        print("Initializing git repository...")
        run_command("git init", "Git initialization")
        run_command("git branch -M main", "Setting main branch")
    
    # Create demo data
    create_demo_data()
    
    # Deploy to GitHub
    if deploy_to_github():
        print("\n🎉 Portfolio deployment successful!")
        print("\n📋 Next steps:")
        print("1. Go to GitHub and make repository public")
        print("2. Add repository to your portfolio website")
        print("3. Consider deploying API to Heroku/Railway for live demo")
        print("4. Add GitHub repository link to your resume")
    else:
        print("\n❌ Deployment failed. Check git configuration.")

if __name__ == "__main__":
    main()