# Personalized LLM Preference Model

A system for training personalized preference models for evaluating LLMs based on user-specific preferences.

## Project Structure

```
preference_model/
├── configs/                 # Configuration files
│   └── experiment_config.yaml
├── data/                   # Raw data storage
├── notebooks/              # Jupyter notebooks for analysis
├── results/                # Model outputs and evaluation results
├── src/                    # Source code
│   └── test_config.py     # Configuration verification script
├── cache/                  # Cached processed data
└── requirements.txt        # Python dependencies
```

## Setup

1. **Activate virtual environment:**
   ```bash
   source ../evals/bin/activate  # Adjust path as needed
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify setup:**
   ```bash
   python src/test_config.py
   ```

## Configuration

All experiment parameters are controlled via `configs/experiment_config.yaml`. Key settings include:

- **Data Sources:** LMSYS Arena conversations and Anthropic HH-RLHF
- **Model Architecture:** Embedding dimension, number of experts, low-rank adaptation
- **Training Parameters:** Learning rates, batch sizes, regularization
- **Experiment Settings:** Minimum votes per user, random seed

## Development Phases

This project follows a 5-step development plan:

1. ✅ **Project Scaffolding & Environment Setup** (Current)
2. **Data Preparation Script**
3. **Global Preference Model Training**
4. **User-Specific Personalization**
5. **Evaluation & Analysis**

## Next Steps

Run the data preparation script to download and process the training datasets.