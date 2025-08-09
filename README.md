## 🎯 Project Objective

The goal is to build and evaluate a system for personalizing LLM rankings, using the official Chatbot Arena ELO scores as the global baseline. We compare:

1. **System A: Elo Offset System** - A simple approach that modifies global ELO scores with user-specific offsets
2. **System B: Personalized Reward Model System** - A sophisticated approach that fine-tunes a pre-trained reward model

##  Codebase

```
├── config.py              # Configuration and parameters
├── data_generation.py     # Synthetic data generation
├── elo_system.py         # Elo Offset system implementation
├── reward_model.py       # Fine-tuned reward model system
├── evaluation.py         # Metrics and evaluation
├── main.py              # Main experiment orchestration
├── prompts/             # Prompt templates
│   ├── prompt_generation.txt
│   ├── preference_evaluation.txt
│   └── ground_truth_ranking.txt
├── cache/               # Cached data and results
└── results/             # Experiment results and plots
```

### Key Components

#### 1. Configuration (`config.py`)
- Model definitions with Arena ELO mappings
- Persona definitions (Coder, Academic, Creative Writer)
- Experiment parameters

#### 2. Data Generation (`data_generation.py`)
- **Prompt Generation**: Uses LLM to generate persona-specific prompts
- **Response Collection**: Gathers responses from all contestant models
- **Preference Evaluation**: Uses LLM to evaluate pairwise preferences

#### 3. Elo Offset System (`elo_system.py`)
- Modifies global ELO scores with user-specific offsets

#### 4. Reward Model System (`reward_model.py`)
- Fine-tunes pre-trained Skywork reward model
- Provides prompt-specific rankings

#### 5. Evaluation (`evaluation.py`)
- Ranking accuracy metrics

## Prerequisites

1. **Ollama**: Install and set up Ollama with the required models
2. **Python**: Python 3.8+ with pip

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd personalized-llm-ranking

# Install dependencies
pip install -r requirements.txt

# Install required Ollama models
ollama pull llama3:8b
ollama pull llama3.1:8b
ollama pull gemma3:4b
ollama pull qwen3:7b
ollama pull phi4:3.8b
ollama pull mistral:7b
```

### Running the Experiment

#### Basic Usage

```bash
# Run experiment with default settings (coder persona)
python main.py

# Run with specific persona
python main.py --persona academic

# Run with custom parameters
python main.py --persona coder --lambda_reg 0.2 --num_prompts 30 --eval_freq 3
```

#### Command Line Options

- `--persona`: Choose persona (coder, academic, creative_writer)
- `--lambda_reg`: Regularization strength for reward model (default: 0.1)
- `--num_prompts`: Number of prompts to generate (default: 25)
- `--eval_freq`: Evaluation frequency (default: 5)
- `--incremental`: Use incremental training (slower but shows learning curves)

### Example Output

```
================================================================================
STARTING PERSONALIZED LLM RANKING EXPERIMENT
Persona: coder
================================================================================

============================================================
GENERATING DATA FOR PERSONA: CODER
============================================================
Using cached prompts for persona: coder
Using cached responses for persona: coder
Using cached preferences for persona: coder

Data Generation Complete:
- Prompts: 25
- Preferences: 300
- Test prompts: 5
- Ground truth rankings: 5

============================================================
TRAINING AND EVALUATION
============================================================
Training on 300 preferences...
Evaluating every 5 preferences...

Processing preferences 1-5...
Evaluation 1:
  Elo Kendall's Tau: 0.234
  Reward Kendall's Tau: 0.156

...

============================================================
FINAL EVALUATION
============================================================

============================================================
FINAL EXPERIMENT RESULTS
============================================================

Persona: coder
Models: llama3.1:8b, gemma3:4b, qwen3:7b, phi4:3.8b, mistral:7b
Lambda (regularization): 0.1

Metric                Elo Offset   Reward Model  Improvement  
------------------------------------------------------------
Kendall Tau           0.456        0.523         +0.067      
Spearman Rho          0.489        0.567         +0.078      
Ranking Accuracy      0.634        0.712         +0.078      
Top 1 Accuracy        0.600        0.800         +0.200      
Top 3 Accuracy        0.733        0.867         +0.134      

Winner: REWARD SYSTEM
🎉 Reward Model system outperforms Elo Offset system!
```

## 🙏 Acknowledgments

- Chatbot Arena for the ELO dataset
- Skywork for the base reward model
- Ollama for the model serving infrastructure 
