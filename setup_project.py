#!/usr/bin/env python3
"""
Setup script for the Personalized LLM Ranking project.
Creates necessary directories and validates the environment.
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(cmd, check=True):
    """Run a shell command and return the result."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=check)
        return result.stdout.strip(), result.stderr.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running command '{cmd}': {e}")
        return None, e.stderr

def check_ollama():
    """Check if Ollama is installed and running."""
    print("Checking Ollama installation...")
    
    # Check if ollama command exists
    stdout, stderr = run_command("ollama --version", check=False)
    if stdout is None:
        print("❌ Ollama is not installed or not in PATH")
        print("Please install Ollama from: https://ollama.ai/")
        return False
    
    print(f"✅ Ollama version: {stdout}")
    
    # Check if ollama is running
    stdout, stderr = run_command("ollama list", check=False)
    if "connection refused" in stderr.lower() or "connect" in stderr.lower():
        print("❌ Ollama is not running")
        print("Please start Ollama: ollama serve")
        return False
    
    print("✅ Ollama is running")
    return True

def check_python_deps():
    """Check if required Python packages are installed."""
    print("Checking Python dependencies...")
    
    required_packages = [
        'torch', 'transformers', 'datasets', 'numpy', 'scipy', 
        'matplotlib', 'seaborn', 'tqdm', 'ollama'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    return True

def create_directories():
    """Create necessary project directories."""
    print("Creating project directories...")
    
    directories = [
        'cache',
        'results',
        'prompts',
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created/verified directory: {directory}")

def check_models():
    """Check if required Ollama models are available."""
    print("Checking Ollama models...")
    
    # Get list of available models
    stdout, stderr = run_command("ollama list", check=False)
    if stdout is None:
        print("❌ Cannot check models - Ollama not responding")
        return False
    
    available_models = [line.split()[0] for line in stdout.split('\n')[1:] if line.strip()]
    
    # Check required models from config
    required_models = [
        'llama3:8b',  # For generation and evaluation
        'llama3.1:8b', 'gemma2:9b', 'qwen2:7b', 'phi3:3.8b', 'mistral:7b'  # Contestant models
    ]
    
    missing_models = []
    for model in required_models:
        if any(model in available for available in available_models):
            print(f"✅ {model} (or variant)")
        else:
            print(f"❌ {model}")
            missing_models.append(model)
    
    if missing_models:
        print(f"\nMissing models: {', '.join(missing_models)}")
        print("Install with: ollama pull <model_name>")
        print("Note: You may need to use alternative model names (e.g., gemma2:9b instead of gemma3:4b)")
        return False
    
    return True

def setup_git():
    """Initialize git repository if not already done."""
    print("Setting up Git repository...")
    
    if os.path.exists('.git'):
        print("✅ Git repository already exists")
        return True
    
    # Initialize git repo
    stdout, stderr = run_command("git init", check=False)
    if stdout is None:
        print("❌ Failed to initialize git repository")
        return False
    
    print("✅ Initialized git repository")
    
    # Add all files
    run_command("git add .", check=False)
    print("✅ Added files to git")
    
    # Initial commit
    run_command('git commit -m "Initial commit: Personalized LLM Ranking Experiment"', check=False)
    print("✅ Created initial commit")
    
    return True

def main():
    """Main setup function."""
    print("🚀 Setting up Personalized LLM Ranking Experiment")
    print("=" * 60)
    
    success = True
    
    # Create directories
    create_directories()
    
    # Check Ollama
    if not check_ollama():
        success = False
    
    # Check Python dependencies
    if not check_python_deps():
        success = False
    
    # Check models (optional - can be installed later)
    models_ok = check_models()
    if not models_ok:
        print("\n⚠️  Some models are missing, but you can install them later")
    
    # Setup git
    setup_git()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Install missing models if any: ollama pull <model_name>")
        print("2. Run the experiment: python main.py --persona coder")
    else:
        print("❌ Setup completed with errors")
        print("Please fix the issues above before running the experiment")
    
    print("\nFor help, see README.md")

if __name__ == "__main__":
    main()