# Project Structure

This document outlines the complete structure of the Personalized LLM Ranking Experiment project.

## 📁 Directory Structure

```
personalized-llm-ranking/
├── 📋 Core Configuration
│   ├── config.py                 # Central configuration and parameters
│   ├── requirements.txt          # Python dependencies
│   └── .gitignore               # Git ignore patterns
│
├── 🔧 Core Modules
│   ├── data_generation.py       # Synthetic data generation
│   ├── elo_system.py            # Elo Offset system implementation
│   ├── reward_model.py          # Fine-tuned reward model system
│   ├── evaluation.py            # Metrics and evaluation
│   └── main.py                  # Main experiment orchestration
│
├── 📝 Prompt Templates
│   ├── prompts/
│   │   ├── prompt_generation.txt      # Template for generating prompts
│   │   ├── preference_evaluation.txt  # Template for pairwise preferences
│   │   └── ground_truth_ranking.txt   # Template for ground truth rankings
│
├── 💾 Data & Results (Generated)
│   ├── cache/                    # Cached data and intermediate results
│   │   ├── arena_elo_scores.json
│   │   ├── generated_prompts_{persona}.json
│   │   ├── model_responses_{persona}.json
│   │   ├── preference_data_{persona}.json
│   │   ├── reward_model_{persona}.pt
│   │   └── elo_state_{persona}.json
│   └── results/                  # Final results and visualizations
│       ├── learning_curves_{persona}.png
│       ├── metric_comparison_{persona}.png
│       └── evaluation_results_{persona}.json
│
├── 🛠️ Setup & Utilities
│   ├── setup_project.py         # Project setup and validation
│   ├── quick_start.sh           # Quick start script
│   └── PROJECT_STRUCTURE.md     # This file
│
└── 📚 Documentation
    └── README.md                 # Main documentation
```

## 🎯 Core Components

### Configuration Layer
- **`config.py`**: Central hub for all configurations
  - Model definitions and Arena ELO mappings
  - Persona definitions (Coder, Academic, Creative Writer)
  - Experiment parameters
  - Path management and caching logic

### Data Processing Layer
- **`data_generation.py`**: Handles all synthetic data generation
  - Prompt generation using LLMs
  - Model response collection
  - Pairwise preference evaluation
  - Ground truth ranking generation
  - Intelligent caching system

### Ranking Systems Layer
- **`elo_system.py`**: Elo Offset system
  - Modifies global ELO scores with user-specific offsets
  - Standard Elo update rules
  - State management and persistence

- **`reward_model.py`**: Personalized Reward Model system
  - Fine-tunes pre-trained Skywork reward model
  - Freezes base model, trains only MLP head
  - Implements rank regularization loss
  - PyTorch-based training pipeline

### Evaluation Layer
- **`evaluation.py`**: Comprehensive evaluation framework
  - Multiple ranking metrics (Kendall's Tau, Spearman's Rho, etc.)
  - Statistical analysis
  - Visualization generation
  - Results comparison

### Orchestration Layer
- **`main.py`**: Main experiment coordinator
  - End-to-end experiment execution
  - Command-line interface
  - Progress monitoring
  - Results aggregation

## 🔄 Data Flow

```
1. Configuration Loading (config.py)
   ↓
2. Prompt Generation (data_generation.py)
   ↓
3. Model Response Collection (data_generation.py)
   ↓
4. Preference Evaluation (data_generation.py)
   ↓
5. System Training
   ├── Elo System Updates (elo_system.py)
   └── Reward Model Training (reward_model.py)
   ↓
6. Evaluation (evaluation.py)
   ↓
7. Results & Visualization (evaluation.py)
```

## 🎨 Prompt Templates

The project uses external prompt templates for better maintainability:

- **`prompt_generation.txt`**: Generates persona-specific prompts
- **`preference_evaluation.txt`**: Evaluates pairwise preferences
- **`ground_truth_ranking.txt`**: Creates ground truth rankings

## 💾 Caching Strategy

The project implements intelligent caching:

- **Config-based invalidation**: Cache invalidated when configuration changes
- **Persona-specific**: Separate caches for each persona
- **Component-level**: Individual caches for prompts, responses, preferences
- **Incremental**: Supports both full and incremental updates

## 🚀 Usage Patterns

### Basic Usage
```bash
python main.py --persona coder
```

### Advanced Usage
```bash
python main.py --persona academic --incremental --lambda_reg 0.2 --num_prompts 30
```

### Development Setup
```bash
./quick_start.sh  # Complete setup
python setup_project.py  # Validation only
```

## 🔧 Extension Points

The modular design enables easy extensions:

1. **New Personas**: Add to `PERSONAS` in `config.py`
2. **New Models**: Add to `DEFAULT_MODELS` in `config.py`
3. **New Metrics**: Extend `Evaluator` class in `evaluation.py`
4. **New Systems**: Implement similar to `EloOffsetSystem`
5. **New Prompts**: Add templates in `prompts/` directory

## 📊 Output Artifacts

Each experiment run produces:

- **Cached Data**: Reusable intermediate results
- **Model Checkpoints**: Trained reward model states
- **Evaluation Metrics**: Comprehensive performance analysis
- **Visualizations**: Learning curves and comparison plots
- **Logs**: Detailed execution logs

This structure ensures reproducibility, maintainability, and extensibility for research purposes.