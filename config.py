"""
Configuration file for personalized LLM ranking experiment.
Contains model mappings, personas, and experiment parameters.
"""

import os
from typing import Dict, List, Any

# Model configurations
DEFAULT_MODELS = {
    'llama3.1:8b': {
        'arena_name': 'Llama-3.1-8B-Instruct',
        'elo': None,  # Will be loaded from dataset
        'ollama_tag': 'llama3.1:8b'
    },
    'gemma3:4b': {
        'arena_name': 'Gemma-3-4B-Instruct',
        'elo': None,
        'ollama_tag': 'gemma3:4b'
    },
    'qwen3:7b': {
        'arena_name': 'Qwen3-7B-Instruct',
        'elo': None,
        'ollama_tag': 'qwen3:7b'
    },
    'phi4:3.8b': {
        'arena_name': 'Phi-4-3.8B-Instruct',
        'elo': None,
        'ollama_tag': 'phi4:3.8b'
    },
    'mistral:7b': {
        'arena_name': 'Mistral-7B-Instruct-v0.3',
        'elo': None,
        'ollama_tag': 'mistral:7b'
    }
}

# Persona definitions
PERSONAS = {
    'coder': {
        'name': 'Professional Software Developer',
        'description': """You are a senior software developer with 8+ years of experience in full-stack development. 
        You specialize in Python, JavaScript, and cloud technologies. You value:
        - Clean, maintainable code with good documentation
        - Performance and efficiency
        - Security best practices
        - Modern development practices (testing, CI/CD)
        - Clear explanations and reasoning
        You prefer responses that are practical, well-structured, and show deep technical understanding.""",
        'expertise': ['Python', 'JavaScript', 'React', 'Node.js', 'AWS', 'Docker', 'Git'],
        'communication_style': 'Direct, technical, prefers code examples'
    },
    'academic': {
        'name': 'Research Scientist',
        'description': """You are a research scientist in machine learning with a PhD from a top university.
        You focus on deep learning, NLP, and AI safety. You value:
        - Rigorous methodology and citations
        - Theoretical understanding
        - Reproducible research
        - Clear mathematical formulations
        - Critical analysis of limitations
        You prefer responses that are thorough, well-referenced, and show deep theoretical knowledge.""",
        'expertise': ['Machine Learning', 'Deep Learning', 'NLP', 'PyTorch', 'TensorFlow', 'Research Methods'],
        'communication_style': 'Academic, detailed, prefers formal explanations'
    },
    'creative_writer': {
        'name': 'Creative Writer and Content Creator',
        'description': """You are a professional creative writer and content creator with experience in fiction, 
        marketing copy, and digital content. You value:
        - Engaging storytelling and narrative flow
        - Emotional resonance and authenticity
        - Creative originality and unique perspectives
        - Clear, accessible language
        - Cultural sensitivity and inclusivity
        You prefer responses that are imaginative, well-crafted, and emotionally engaging.""",
        'expertise': ['Creative Writing', 'Content Marketing', 'Storytelling', 'SEO', 'Social Media'],
        'communication_style': 'Creative, engaging, prefers narrative approaches'
    }
}

# Experiment parameters
EXPERIMENT_CONFIG = {
    'num_prompts': 25,  # Number of prompts to generate per persona
    'train_test_split': 0.8,  # 80% for training, 20% for testing
    'evaluation_frequency': 5,  # Evaluate every N preference pairs
    'lambda_reg': 0.1,  # Regularization strength for reward model
    'learning_rate': 1e-4,  # Learning rate for reward model fine-tuning
    'batch_size': 4,  # Batch size for training
    'max_epochs': 10,  # Maximum training epochs
    'random_seed': 42,  # Random seed for reproducibility
    'incremental_training': False,  # Use incremental training (True) or single training (False)
}

# Model paths and identifiers
MODEL_PATHS = {
    'generator_model': 'llama3:8b',  # For generating prompts
    'evaluator_model': 'llama3:8b',  # For evaluating preferences
    'base_reward_model': 'Skywork/Skywork-Reward-V2-Llama-3.2-1B',  # Base reward model for fine-tuning
}

# Data paths
DATA_PATHS = {
    'cache_dir': 'cache',
    'results_dir': 'results',
    'prompts_dir': 'prompts',
    'arena_dataset': 'mathewhe/chatbot-arena-elo',
}

# Ensure directories exist
for path in DATA_PATHS.values():
    if path not in ['mathewhe/chatbot-arena-elo']:
        os.makedirs(path, exist_ok=True)

# Cache file names
CACHE_FILES = {
    'arena_elo_scores': 'arena_elo_scores.json',
    'generated_prompts': 'generated_prompts_{persona}.json',
    'model_responses': 'model_responses_{persona}.json',
    'preference_data': 'preference_data_{persona}.json',
    'training_history': 'training_history_{persona}.json',
}

def get_cache_file_path(filename: str, persona: str = None) -> str:
    """Get the full path for a cache file."""
    if persona and '{persona}' in filename:
        filename = filename.format(persona=persona)
    return os.path.join(DATA_PATHS['cache_dir'], filename)

def get_config_hash() -> str:
    """Generate a hash of the current configuration for cache invalidation."""
    import hashlib
    import json
    
    config_data = {
        'models': DEFAULT_MODELS,
        'personas': {k: v['description'] for k, v in PERSONAS.items()},
        'experiment': EXPERIMENT_CONFIG,
        'model_paths': MODEL_PATHS
    }
    
    config_str = json.dumps(config_data, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:8] 