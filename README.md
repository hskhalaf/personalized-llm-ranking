# Personalized LLM Ranking Experiment

This repository implements an experimental framework for comparing two approaches to personalized LLM ranking: **Elo Offset System** vs **Fine-tuned Reward Model System**.

## 🎯 Project Objective

The goal is to build and evaluate two systems for personalizing LLM rankings, using the official Chatbot Arena ELO scores as the global baseline. We compare:

1. **System A: Elo Offset System** - A simple approach that modifies global ELO scores with user-specific offsets
2. **System B: Personalized Reward Model System** - A sophisticated approach that fine-tunes a pre-trained reward model

## 🧪 Core Hypotheses

1. **Sample Efficiency Hypothesis**: The fine-tuned Reward Model will learn user preferences more accurately with fewer examples than the Elo Offset system.
2. **Generalization Hypothesis**: The fine-tuned Reward Model will be better at ranking unseen models by analyzing response content, outperforming the Elo Offset system which only uses static global ELO scores.

## 🏗️ Architecture

### Modular Design

The codebase is organized into clear, modular components:

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
- Cache management

#### 2. Data Generation (`data_generation.py`)
- **Prompt Generation**: Uses LLM to generate persona-specific prompts
- **Response Collection**: Gathers responses from all contestant models
- **Preference Evaluation**: Uses LLM to evaluate pairwise preferences
- **Caching**: Intelligent caching to avoid regeneration

#### 3. Elo Offset System (`elo_system.py`)
- Modifies global ELO scores with user-specific offsets
- Uses standard Elo update rules
- Provides personalized rankings

#### 4. Reward Model System (`reward_model.py`)
- Fine-tunes pre-trained Skywork reward model
- Freezes base model, trains only MLP head
- Implements rank regularization loss
- Provides prompt-specific rankings

#### 5. Evaluation (`evaluation.py`)
- Kendall's Tau correlation
- Spearman's Rho correlation
- Ranking accuracy metrics
- Learning curve visualization

## 🚀 Quick Start

### Prerequisites

1. **Ollama**: Install and set up Ollama with the required models
2. **Python**: Python 3.8+ with pip
3. **GPU**: Recommended for reward model training (CUDA-compatible)

### Installation

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

## 📊 Results and Analysis

### Metrics

The experiment evaluates both systems using:

1. **Kendall's Tau**: Rank correlation coefficient
2. **Spearman's Rho**: Rank correlation coefficient
3. **Ranking Accuracy**: Fraction of correctly ordered pairs
4. **Top-k Accuracy**: Accuracy for top-k models

### Visualizations

The experiment generates:

1. **Learning Curves**: Performance vs. training steps
2. **Metric Comparison**: Final performance comparison
3. **Training History**: Detailed training logs

### Caching System

The system implements intelligent caching:

- **Config-based invalidation**: Cache is invalidated when configuration changes
- **Persona-specific caching**: Separate caches for each persona
- **Structured data storage**: JSON format for easy inspection

## 🔧 Configuration

### Models

Default models (configurable in `config.py`):

```python
DEFAULT_MODELS = {
    'llama3.1:8b': {
        'arena_name': 'Llama-3.1-8B-Instruct',
        'elo': None,  # Loaded from dataset
        'ollama_tag': 'llama3.1:8b'
    },
    'gemma3:4b': {
        'arena_name': 'Gemma-3-4B-Instruct',
        'elo': None,
        'ollama_tag': 'gemma3:4b'
    },
    # ... more models
}
```

### Personas

Three detailed personas are included:

1. **Coder**: Professional software developer
2. **Academic**: Research scientist in ML
3. **Creative Writer**: Content creator and writer

### Experiment Parameters

```python
EXPERIMENT_CONFIG = {
    'num_prompts': 25,
    'train_test_split': 0.8,
    'evaluation_frequency': 5,
    'lambda_reg': 0.1,
    'learning_rate': 1e-4,
    'batch_size': 4,
    'max_epochs': 10,
    'random_seed': 42,
}
```

## 🧪 Experimental Design

### Data Generation Process

1. **Prompt Generation**: LLM generates diverse, persona-specific prompts
2. **Response Collection**: All models respond to all prompts
3. **Preference Evaluation**: LLM evaluates pairwise preferences
4. **Ground Truth**: LLM generates ground truth rankings for test set

### Training Process

1. **Elo System**: Updates offsets using standard Elo rules
2. **Reward Model**: Fine-tunes with pairwise loss + rank regularization
   - **Single Training** (default): Trains once on all training data (faster)
   - **Incremental Training** (optional): Retrains on growing dataset at each evaluation step
     - Simulates online learning scenario where model continuously improves
     - More computationally intensive but provides better learning curves
     - Use `--incremental` flag to enable
3. **Evaluation**: Both systems evaluated on test prompts
4. **Comparison**: Metrics compared to determine winner

### Evaluation Protocol

- **Frequency**: Evaluate every N preference pairs
- **Metrics**: Kendall's Tau, Spearman's Rho, ranking accuracy
- **Ground Truth**: LLM-generated rankings for test prompts
- **Comparison**: Direct comparison between systems

## 📁 File Structure

```
personalized-llm-ranking/
├── config.py                 # Configuration and parameters
├── data_generation.py        # Data generation module
├── elo_system.py            # Elo Offset system
├── reward_model.py          # Reward model system
├── evaluation.py            # Evaluation and metrics
├── main.py                  # Main experiment orchestration
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── prompts/                # Prompt templates
│   ├── prompt_generation.txt
│   ├── preference_evaluation.txt
│   └── ground_truth_ranking.txt
├── cache/                  # Cached data
│   ├── arena_elo_scores.json
│   ├── generated_prompts_coder.json
│   ├── model_responses_coder.json
│   ├── preference_data_coder.json
│   └── ...
└── results/                # Experiment results
    ├── learning_curves_coder.png
    ├── metric_comparison_coder.png
    └── evaluation_results_coder.json
```

## 🔍 Troubleshooting

### Common Issues

1. **Ollama Connection Error**: Ensure Ollama is running and models are installed
2. **CUDA Out of Memory**: Reduce batch size or use CPU
3. **Model Not Found**: Check model names in config and Ollama installation
4. **Cache Issues**: Delete cache directory to regenerate data

### Model Verification

Before running the experiment, verify that your Ollama models match the configuration:

```bash
# Check installed models
ollama list

# Verify model tags match config.py
# Update config.py if needed to match your installed models
```

**Important**: Some model tags in the default configuration (like `gemma3:4b`) may not exist on Ollama. You may need to use alternative models like `gemma2:9b` or check the [Ollama library](https://ollama.ai/library) for available models.

### Debug Mode

Add debug prints by modifying the logging level in the code.

## ⚠️ Limitations

### LLM Positional Bias

The ground truth ranking generation uses a single prompt to rank all models simultaneously. This approach may introduce positional bias, where LLMs favor items at the beginning or end of the list. For more robust results, consider implementing a round-robin tournament with pairwise comparisons, though this would significantly increase computation time.

### Synthetic Data

This experiment uses synthetic data generated by LLMs rather than human preferences. While this enables controlled experimentation, the results may not fully generalize to real human preferences. Future work should validate findings with human preference data.

### Model Availability

The experiment depends on specific models being available in both Ollama and the Chatbot Arena dataset. Model availability may change over time, requiring configuration updates.

## 📈 Future Work

1. **More Personas**: Add diverse user personas
2. **Model Diversity**: Test with different model families
3. **Advanced Regularization**: Explore different regularization techniques
4. **Real User Data**: Validate with human preference data
5. **Ablation Studies**: Analyze component contributions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Chatbot Arena for the ELO dataset
- Skywork for the base reward model
- Ollama for the model serving infrastructure 