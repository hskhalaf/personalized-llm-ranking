#!/usr/bin/env python3
"""
PyTorch Models for Personalized LLM Preference Learning

This module implements:
1. MoEPreferenceModel: Main mixture-of-experts preference model
2. SingleVectorModel: Baseline with K=1 expert
3. BradleyTerryModel: Simple Bradley-Terry baseline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class MoEPreferenceModel(nn.Module):
    """
    Mixture of Experts Preference Model
    
    Computes utility score S(u, m, x) where:
    - u: user (represented by personalization matrix A_u)
    - m: model (represented by model_id)
    - x: prompt (represented by prompt_embedding)
    
    Architecture:
    - v_m: Model embeddings (num_models, embedding_dim)
    - mu_G: Global expert centers (num_experts_K, embedding_dim)
    - B: Low-rank transformation matrix (embedding_dim, low_rank_r)
    - A_u: User-specific personalization (num_experts_K, low_rank_r) - input parameter
    
    Forward pass:
    1. delta_u = A_u @ B.T  -> (K, embedding_dim)
    2. For each expert k: s_k = ((mu_G[k] + delta_u[k]) * prompt_embedding).T @ v_m
    3. S = log_sum_exp(s_k) over all experts k
    """
    
    def __init__(self, num_models: int, embedding_dim: int, num_experts_K: int, low_rank_r: int):
        """
        Initialize the MoE Preference Model.
        
        Args:
            num_models: Number of LLM models in vocabulary
            embedding_dim: Dimension of prompt embeddings (e.g., 384)
            num_experts_K: Number of experts in the mixture
            low_rank_r: Rank of the low-rank adaptation
        """
        super().__init__()
        
        # Store hyperparameters
        self.num_models = num_models
        self.embedding_dim = embedding_dim
        self.num_experts_K = num_experts_K
        self.low_rank_r = low_rank_r
        
        # Model embeddings: v_m for each model m
        self.v_m = nn.Embedding(num_models, embedding_dim)
        
        # Global expert centers: mu_G[k] for each expert k
        self.mu_G = nn.Parameter(torch.randn(num_experts_K, embedding_dim))
        
        # Low-rank transformation matrix: B
        self.B = nn.Parameter(torch.randn(embedding_dim, low_rank_r))
        
        # Initialize parameters
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize model parameters with reasonable values."""
        # Initialize model embeddings with small random values
        nn.init.normal_(self.v_m.weight, mean=0.0, std=0.1)
        
        # Initialize global expert centers with small random values
        nn.init.normal_(self.mu_G, mean=0.0, std=0.1)
        
        # Initialize low-rank matrix with small random values
        nn.init.normal_(self.B, mean=0.0, std=0.1)
        
    def forward(self, prompt_embedding: torch.Tensor, model_id: torch.Tensor, A_u: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute utility score S(u, m, x).
        
        Args:
            prompt_embedding: Pre-computed prompt embedding, shape (embedding_dim,) or (batch_size, embedding_dim)
            model_id: Model ID(s), shape () or (batch_size,)
            A_u: User personalization matrix, shape (num_experts_K, low_rank_r)
            
        Returns:
            Utility scores S(u, m, x), shape () or (batch_size,)
        """
        # Ensure tensors are on the same device
        device = self.v_m.weight.device
        prompt_embedding = prompt_embedding.to(device)
        model_id = model_id.to(device)
        A_u = A_u.to(device)
        
        # Handle batch vs single example
        if prompt_embedding.dim() == 1:
            # Single example
            batch_size = 1
            prompt_embedding = prompt_embedding.unsqueeze(0)  # (1, embedding_dim)
            model_id = model_id.unsqueeze(0) if model_id.dim() == 0 else model_id  # (1,)
            single_example = True
        else:
            # Batch of examples
            batch_size = prompt_embedding.shape[0]
            single_example = False
            
        # Get model embeddings: v_m
        v_m = self.v_m(model_id)  # (batch_size, embedding_dim)
        
        # Compute user-specific expert adjustments: delta_u = A_u @ B.T
        delta_u = torch.matmul(A_u, self.B.T)  # (num_experts_K, embedding_dim)
        
        # Compute expert scores s_k for each expert k
        expert_scores = []
        for k in range(self.num_experts_K):
            # Adjusted expert center: mu_G[k] + delta_u[k]
            adjusted_expert = self.mu_G[k] + delta_u[k]  # (embedding_dim,)
            
            # Element-wise product: (adjusted_expert * prompt_embedding)
            # prompt_embedding: (batch_size, embedding_dim)
            # adjusted_expert: (embedding_dim,)
            element_wise_product = adjusted_expert.unsqueeze(0) * prompt_embedding  # (batch_size, embedding_dim)
            
            # Score: (element_wise_product).T @ v_m
            # This is equivalent to: sum(element_wise_product * v_m, dim=-1)
            s_k = torch.sum(element_wise_product * v_m, dim=-1)  # (batch_size,)
            expert_scores.append(s_k)
            
        # Stack expert scores: (num_experts_K, batch_size)
        expert_scores = torch.stack(expert_scores, dim=0)
        
        # Compute log-sum-exp over experts (numerically stable)
        S = torch.logsumexp(expert_scores, dim=0)  # (batch_size,)
        
        # Return single value if input was single example
        if single_example:
            S = S.squeeze(0)
            
        return S
    
    @classmethod
    def from_config(cls, config_path: str, num_models: int):
        """Create model from configuration file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        return cls(
            num_models=num_models,
            embedding_dim=config['embedding_dim'],
            num_experts_K=config['num_experts_K'],
            low_rank_r=config['low_rank_r']
        )


class SingleVectorModel(nn.Module):
    """
    Single Vector Baseline Model (K=1 case)
    
    This is a simplified version of the MoE model with only one expert.
    Equivalent to MoEPreferenceModel with num_experts_K=1.
    """
    
    def __init__(self, num_models: int, embedding_dim: int, low_rank_r: int):
        """
        Initialize the Single Vector Model.
        
        Args:
            num_models: Number of LLM models in vocabulary
            embedding_dim: Dimension of prompt embeddings
            low_rank_r: Rank of the low-rank adaptation
        """
        super().__init__()
        
        # Use MoE model with K=1
        self.moe_model = MoEPreferenceModel(
            num_models=num_models,
            embedding_dim=embedding_dim,
            num_experts_K=1,  # Single expert
            low_rank_r=low_rank_r
        )
        
    def forward(self, prompt_embedding: torch.Tensor, model_id: torch.Tensor, A_u: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using single expert.
        
        Args:
            prompt_embedding: Pre-computed prompt embedding
            model_id: Model ID(s)
            A_u: User personalization matrix, shape (1, low_rank_r)
            
        Returns:
            Utility scores
        """
        return self.moe_model(prompt_embedding, model_id, A_u)
    
    @classmethod
    def from_config(cls, config_path: str, num_models: int):
        """Create model from configuration file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        return cls(
            num_models=num_models,
            embedding_dim=config['embedding_dim'],
            low_rank_r=config['low_rank_r']
        )


class BradleyTerryModel(nn.Module):
    """
    Bradley-Terry Baseline Model
    
    Simple preference model: S(u, m, x) = w_u.T @ v_m
    
    This model ignores the prompt and only considers user preferences
    for different models directly.
    """
    
    def __init__(self, num_models: int, embedding_dim: int):
        """
        Initialize the Bradley-Terry Model.
        
        Args:
            num_models: Number of LLM models in vocabulary
            embedding_dim: Dimension for consistency (user and model embeddings)
        """
        super().__init__()
        
        self.num_models = num_models
        self.embedding_dim = embedding_dim
        
        # Model embeddings: v_m for each model m
        self.v_m = nn.Embedding(num_models, embedding_dim)
        
        # Initialize parameters
        nn.init.normal_(self.v_m.weight, mean=0.0, std=0.1)
        
    def forward(self, prompt_embedding: torch.Tensor, model_id: torch.Tensor, w_u: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute Bradley-Terry score.
        
        Args:
            prompt_embedding: Pre-computed prompt embedding (ignored in this model)
            model_id: Model ID(s)
            w_u: User preference vector, shape (embedding_dim,)
            
        Returns:
            Utility scores S(u, m, x) = w_u.T @ v_m
        """
        # Ensure tensors are on the same device
        device = self.v_m.weight.device
        model_id = model_id.to(device)
        w_u = w_u.to(device)
        
        # Handle batch vs single example
        if model_id.dim() == 0:
            # Single example
            model_id = model_id.unsqueeze(0)
            single_example = True
        else:
            single_example = False
            
        # Get model embeddings: v_m
        v_m = self.v_m(model_id)  # (batch_size, embedding_dim)
        
        # Compute scores: w_u.T @ v_m
        # w_u: (embedding_dim,)
        # v_m: (batch_size, embedding_dim)
        scores = torch.matmul(v_m, w_u)  # (batch_size,)
        
        # Return single value if input was single example
        if single_example:
            scores = scores.squeeze(0)
            
        return scores
    
    @classmethod
    def from_config(cls, config_path: str, num_models: int):
        """Create model from configuration file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        return cls(
            num_models=num_models,
            embedding_dim=config['embedding_dim']
        )


def create_random_user_params(model_type: str, config_path: str) -> torch.Tensor:
    """
    Create random user parameters for testing.
    
    Args:
        model_type: Type of model ('moe', 'single', 'bradley_terry')
        config_path: Path to configuration file
        
    Returns:
        Random user parameters tensor
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if model_type == 'moe':
        # A_u matrix: (num_experts_K, low_rank_r)
        return torch.randn(config['num_experts_K'], config['low_rank_r'])
    elif model_type == 'single':
        # A_u matrix: (1, low_rank_r)
        return torch.randn(1, config['low_rank_r'])
    elif model_type == 'bradley_terry':
        # w_u vector: (embedding_dim,)
        return torch.randn(config['embedding_dim'])
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Simple test when run as script
    print("Testing model implementations...")
    
    # Test parameters
    num_models = 20
    embedding_dim = 384
    num_experts_K = 2
    low_rank_r = 16
    
    # Create test inputs
    prompt_embedding = torch.randn(embedding_dim)
    model_id = torch.tensor(5)
    
    # Test MoE Model
    print("\n1. Testing MoEPreferenceModel...")
    moe_model = MoEPreferenceModel(num_models, embedding_dim, num_experts_K, low_rank_r)
    A_u = torch.randn(num_experts_K, low_rank_r)
    
    score = moe_model(prompt_embedding, model_id, A_u)
    print(f"   MoE score: {score.item():.4f}")
    
    # Test Single Vector Model
    print("\n2. Testing SingleVectorModel...")
    single_model = SingleVectorModel(num_models, embedding_dim, low_rank_r)
    A_u_single = torch.randn(1, low_rank_r)
    
    score = single_model(prompt_embedding, model_id, A_u_single)
    print(f"   Single vector score: {score.item():.4f}")
    
    # Test Bradley-Terry Model
    print("\n3. Testing BradleyTerryModel...")
    bt_model = BradleyTerryModel(num_models, embedding_dim)
    w_u = torch.randn(embedding_dim)
    
    score = bt_model(prompt_embedding, model_id, w_u)
    print(f"   Bradley-Terry score: {score.item():.4f}")
    
    print("\n✅ All models working correctly!")