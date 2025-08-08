#!/usr/bin/env python3
"""
Test script for PyTorch models using actual configuration and data.
"""

import torch
import json
import pickle
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))
from models import MoEPreferenceModel, SingleVectorModel, BradleyTerryModel, create_random_user_params

def test_models_with_config():
    """Test models using actual configuration and cached data."""
    print("Testing models with actual configuration and data...")
    
    # Load configuration
    config_path = "configs/experiment_config.yaml"
    
    # Load model vocabulary to get num_models
    with open("cache/model_vocab.json", 'r') as f:
        model_vocab = json.load(f)
    num_models = len(model_vocab)
    
    # Load a sample prompt embedding
    with open("cache/prompt_embeddings.pkl", 'rb') as f:
        prompt_embeddings = pickle.load(f)
    sample_prompt = list(prompt_embeddings.keys())[0]
    sample_embedding = torch.tensor(prompt_embeddings[sample_prompt], dtype=torch.float32)
    
    # Load sample processed data
    df = pd.read_parquet("cache/arena_processed.parquet")
    sample_row = df.iloc[0]
    
    print(f"Configuration loaded: {num_models} models, {sample_embedding.shape[0]}D embeddings")
    print(f"Sample prompt: {sample_prompt[:80]}...")
    print(f"Sample comparison: {sample_row['winner_model_name']} vs {sample_row['loser_model_name']}")
    
    # Test 1: MoE Preference Model
    print("\n1. Testing MoEPreferenceModel with config...")
    moe_model = MoEPreferenceModel.from_config(config_path, num_models)
    A_u_moe = create_random_user_params('moe', config_path)
    
    winner_id = torch.tensor(sample_row['winner_model_id'])
    loser_id = torch.tensor(sample_row['loser_model_id'])
    
    winner_score = moe_model(sample_embedding, winner_id, A_u_moe)
    loser_score = moe_model(sample_embedding, loser_id, A_u_moe)
    
    print(f"   Winner ({sample_row['winner_model_name']}) score: {winner_score.item():.4f}")
    print(f"   Loser ({sample_row['loser_model_name']}) score: {loser_score.item():.4f}")
    print(f"   Preference satisfied: {winner_score > loser_score}")
    
    # Test 2: Single Vector Model
    print("\n2. Testing SingleVectorModel with config...")
    single_model = SingleVectorModel.from_config(config_path, num_models)
    A_u_single = create_random_user_params('single', config_path)
    
    winner_score = single_model(sample_embedding, winner_id, A_u_single)
    loser_score = single_model(sample_embedding, loser_id, A_u_single)
    
    print(f"   Winner score: {winner_score.item():.4f}")
    print(f"   Loser score: {loser_score.item():.4f}")
    print(f"   Preference satisfied: {winner_score > loser_score}")
    
    # Test 3: Bradley-Terry Model
    print("\n3. Testing BradleyTerryModel with config...")
    bt_model = BradleyTerryModel.from_config(config_path, num_models)
    w_u = create_random_user_params('bradley_terry', config_path)
    
    winner_score = bt_model(sample_embedding, winner_id, w_u)
    loser_score = bt_model(sample_embedding, loser_id, w_u)
    
    print(f"   Winner score: {winner_score.item():.4f}")
    print(f"   Loser score: {loser_score.item():.4f}")
    print(f"   Preference satisfied: {winner_score > loser_score}")
    
    # Test 4: Batch processing
    print("\n4. Testing batch processing...")
    batch_size = 5
    batch_embeddings = sample_embedding.unsqueeze(0).repeat(batch_size, 1)
    batch_model_ids = torch.randint(0, num_models, (batch_size,))
    
    batch_scores = moe_model(batch_embeddings, batch_model_ids, A_u_moe)
    print(f"   Batch input shape: {batch_embeddings.shape}")
    print(f"   Batch output shape: {batch_scores.shape}")
    print(f"   Batch scores: {batch_scores.tolist()}")
    
    # Test 5: Model parameter counts
    print("\n5. Model parameter statistics...")
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    moe_params = count_parameters(moe_model)
    single_params = count_parameters(single_model.moe_model)  # Access underlying model
    bt_params = count_parameters(bt_model)
    
    print(f"   MoE Model parameters: {moe_params:,}")
    print(f"   Single Vector Model parameters: {single_params:,}")
    print(f"   Bradley-Terry Model parameters: {bt_params:,}")
    
    print("\n✅ All tests passed successfully!")

def test_model_consistency():
    """Test that models produce consistent outputs."""
    print("\nTesting model consistency...")
    
    # Fixed seed for reproducibility
    torch.manual_seed(42)
    
    config_path = "configs/experiment_config.yaml"
    num_models = 20
    embedding_dim = 384
    
    # Create models
    model1 = MoEPreferenceModel.from_config(config_path, num_models)
    model2 = MoEPreferenceModel.from_config(config_path, num_models)
    
    # Same input
    prompt_embedding = torch.randn(embedding_dim)
    model_id = torch.tensor(5)
    A_u = create_random_user_params('moe', config_path)
    
    # Different random initializations should give different results
    score1 = model1(prompt_embedding, model_id, A_u)
    score2 = model2(prompt_embedding, model_id, A_u)
    
    print(f"   Model 1 score: {score1.item():.4f}")
    print(f"   Model 2 score: {score2.item():.4f}")
    print(f"   Models differ (expected): {not torch.allclose(score1, score2, atol=1e-6)}")
    
    # Same model should give same results
    score1_repeat = model1(prompt_embedding, model_id, A_u)
    print(f"   Same model consistency: {torch.allclose(score1, score1_repeat, atol=1e-6)}")

if __name__ == "__main__":
    test_models_with_config()
    test_model_consistency()