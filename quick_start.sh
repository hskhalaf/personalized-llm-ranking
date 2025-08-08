#!/bin/bash
# Quick start script for Personalized LLM Ranking Experiment

echo "🚀 Personalized LLM Ranking Experiment - Quick Start"
echo "=================================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed"
    exit 1
fi

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Run setup script
echo "⚙️  Running setup script..."
python3 setup_project.py

# Check if setup was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Setup completed!"
    echo ""
    echo "🎯 Ready to run experiments:"
    echo "   python main.py --persona coder"
    echo "   python main.py --persona academic --incremental"
    echo "   python main.py --persona creative_writer --lambda_reg 0.2"
    echo ""
    echo "📖 For more options, see: python main.py --help"
    echo "📚 For detailed documentation, see: README.md"
else
    echo "❌ Setup failed. Please check the output above."
fi