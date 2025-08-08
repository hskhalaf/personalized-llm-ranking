"""
Personalized Reward Model System for LLM ranking.
Fine-tunes a pre-trained reward model with user preferences.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import Dict, List, Tuple, Any
import json
import os
from tqdm import tqdm

from config import DEFAULT_MODELS, EXPERIMENT_CONFIG, MODEL_PATHS, get_cache_file_path


class RewardModelDataset(Dataset):
    """Dataset for training the reward model."""
    
    def __init__(self, preferences: List[Dict], responses: Dict, tokenizer, max_length: int = 512):
        """
        Initialize the dataset.
        
        Args:
            preferences: List of preference dictionaries
            responses: Dictionary of model responses
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
        """
        self.preferences = preferences
        self.responses = responses
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Prepare data
        self.data = self._prepare_data()
    
    def _prepare_data(self) -> List[Dict]:
        """Prepare the data for training."""
        data = []
        
        for pref in self.preferences:
            prompt_id = pref['prompt_id']
            prompt_text = pref['prompt_text']
            winner = pref['winner']
            loser = pref['loser']
            
            # Get responses
            if prompt_id in self.responses:
                prompt_responses = self.responses[prompt_id]
                if winner in prompt_responses and loser in prompt_responses:
                    winner_response = prompt_responses[winner]
                    loser_response = prompt_responses[loser]
                    
                    # Format as conversation
                    winner_conversation = self._format_conversation(prompt_text, winner_response)
                    loser_conversation = self._format_conversation(prompt_text, loser_response)
                    
                    data.append({
                        'winner_conversation': winner_conversation,
                        'loser_conversation': loser_conversation,
                        'winner': winner,
                        'loser': loser,
                        'prompt_id': prompt_id
                    })
        
        return data
    
    def _format_conversation(self, prompt: str, response: str) -> str:
        """Format prompt and response as a conversation."""
        # Use the tokenizer's chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template'):
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response}
            ]
            return self.tokenizer.apply_chat_template(messages, tokenize=False)
        else:
            # Fallback to simple formatting
            return f"User: {prompt}\nAssistant: {response}"
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        
        # Tokenize conversations
        winner_tokens = self.tokenizer(
            item['winner_conversation'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        loser_tokens = self.tokenizer(
            item['loser_conversation'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'winner_input_ids': winner_tokens['input_ids'].squeeze(0),
            'winner_attention_mask': winner_tokens['attention_mask'].squeeze(0),
            'loser_input_ids': loser_tokens['input_ids'].squeeze(0),
            'loser_attention_mask': loser_tokens['attention_mask'].squeeze(0),
            'winner': item['winner'],
            'loser': item['loser'],
            'prompt_id': item['prompt_id']
        }


class PersonalizedRewardModel(nn.Module):
    """Personalized reward model with trainable head on frozen base model."""
    
    def __init__(self, base_model_name: str, models: Dict = None):
        """
        Initialize the personalized reward model.
        
        Args:
            base_model_name: Name/path of the base reward model
            models: Dictionary of models for regularization
        """
        super().__init__()
        
        self.models = models or DEFAULT_MODELS
        self.base_model_name = base_model_name
        
        # Load base model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.base_model = AutoModel.from_pretrained(base_model_name)
        
        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Get hidden size
        hidden_size = self.base_model.config.hidden_size
        
        # Add trainable MLP head
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1)
        )
        
        # Initialize head weights
        self._init_head_weights()
    
    def _init_head_weights(self):
        """Initialize the reward head weights."""
        for module in self.reward_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass through the model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            
        Returns:
            Reward scores
        """
        # Get base model outputs
        with torch.no_grad():
            outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Get the last hidden state
        hidden_states = outputs.last_hidden_state
        
        # Pool the hidden states (mean pooling)
        if attention_mask is not None:
            # Masked mean pooling
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            pooled_output = sum_hidden / sum_mask
        else:
            # Simple mean pooling
            pooled_output = torch.mean(hidden_states, dim=1)
        
        # Pass through reward head
        rewards = self.reward_head(pooled_output)
        
        return rewards.squeeze(-1)
    
    def get_reward(self, conversation: str) -> float:
        """
        Get reward score for a conversation.
        
        Args:
            conversation: Formatted conversation string
            
        Returns:
            Reward score
        """
        self.eval()
        with torch.no_grad():
            tokens = self.tokenizer(
                conversation,
                truncation=True,
                padding=True,
                return_tensors='pt'
            )
            
            if next(self.parameters()).is_cuda:
                tokens = {k: v.cuda() for k, v in tokens.items()}
            
            reward = self.forward(**tokens)
            return reward.item()


class PersonalizedRewardSystem:
    """System for training and using personalized reward models."""
    
    def __init__(self, models: Dict = None, lambda_reg: float = None):
        """
        Initialize the personalized reward system.
        
        Args:
            models: Dictionary of models
            lambda_reg: Regularization strength
        """
        self.models = models or DEFAULT_MODELS
        self.lambda_reg = lambda_reg or EXPERIMENT_CONFIG['lambda_reg']
        
        # Initialize reward model
        self.reward_model = None
        self.tokenizer = None
        
        # Training history
        self.training_history = []
    
    def initialize_model(self, base_model_name: str = None):
        """Initialize the reward model."""
        base_model_name = base_model_name or MODEL_PATHS['base_reward_model']
        
        print(f"Loading base reward model: {base_model_name}")
        self.reward_model = PersonalizedRewardModel(base_model_name, self.models)
        self.tokenizer = self.reward_model.tokenizer
        
        # Move to GPU if available
        if torch.cuda.is_available():
            self.reward_model = self.reward_model.cuda()
            print("Model moved to GPU")
    
    def _pairwise_loss(self, winner_scores, loser_scores):
        """Calculate pairwise preference loss."""
        return -F.logsigmoid(winner_scores - loser_scores).mean()
    
    def _regularization_loss(self, winner_scores, loser_scores, winner, loser):
        """Calculate regularization loss based on global ELO ranking."""
        # Get global ELO scores
        winner_elo = self.models[winner]['elo'] if self.models[winner]['elo'] else 1200.0
        loser_elo = self.models[loser]['elo'] if self.models[loser]['elo'] else 1200.0
        
        # If global ranking is inverted (loser has higher ELO), apply penalty
        if winner_elo < loser_elo:
            return -F.logsigmoid(loser_scores - winner_scores).mean()
        else:
            return torch.tensor(0.0, device=winner_scores.device)
    
    def train(self, preferences: List[Dict], responses: Dict, 
              learning_rate: float = None, batch_size: int = None, 
              max_epochs: int = None, persona: str = None):
        """
        Train the personalized reward model.
        
        Args:
            preferences: List of preference dictionaries
            responses: Dictionary of model responses
            learning_rate: Learning rate for training
            batch_size: Batch size for training
            max_epochs: Maximum number of epochs
            persona: Persona name for saving
        """
        if self.reward_model is None:
            self.initialize_model()
        
        learning_rate = learning_rate or EXPERIMENT_CONFIG['learning_rate']
        batch_size = batch_size or EXPERIMENT_CONFIG['batch_size']
        max_epochs = max_epochs or EXPERIMENT_CONFIG['max_epochs']
        
        # Create dataset and dataloader
        dataset = RewardModelDataset(preferences, responses, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(self.reward_model.reward_head.parameters(), lr=learning_rate)
        
        # Training loop
        self.reward_model.train()
        
        for epoch in range(max_epochs):
            epoch_losses = []
            
            for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{max_epochs}"):
                # Move batch to device
                if torch.cuda.is_available():
                    batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                
                # Forward pass
                winner_scores = self.reward_model(
                    batch['winner_input_ids'],
                    batch['winner_attention_mask']
                )
                
                loser_scores = self.reward_model(
                    batch['loser_input_ids'],
                    batch['loser_attention_mask']
                )
                
                # Calculate losses
                pairwise_loss = self._pairwise_loss(winner_scores, loser_scores)
                
                # Calculate regularization loss for each pair
                reg_losses = []
                for i in range(len(batch['winner'])):
                    reg_loss = self._regularization_loss(
                        winner_scores[i:i+1], loser_scores[i:i+1],
                        batch['winner'][i], batch['loser'][i]
                    )
                    reg_losses.append(reg_loss)
                
                reg_loss = torch.stack(reg_losses).mean()
                
                # Total loss
                total_loss = pairwise_loss + self.lambda_reg * reg_loss
                
                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                epoch_losses.append({
                    'total_loss': total_loss.item(),
                    'pairwise_loss': pairwise_loss.item(),
                    'reg_loss': reg_loss.item()
                })
            
            # Log epoch results
            avg_losses = {
                'total_loss': np.mean([l['total_loss'] for l in epoch_losses]),
                'pairwise_loss': np.mean([l['pairwise_loss'] for l in epoch_losses]),
                'reg_loss': np.mean([l['reg_loss'] for l in epoch_losses])
            }
            
            self.training_history.append({
                'epoch': epoch + 1,
                'losses': avg_losses
            })
            
            print(f"Epoch {epoch+1}: Total Loss: {avg_losses['total_loss']:.4f}, "
                  f"Pairwise: {avg_losses['pairwise_loss']:.4f}, "
                  f"Reg: {avg_losses['reg_loss']:.4f}")
        
        # Save training history
        if persona:
            self.save_training_history(persona)
    
    def get_ranking(self, prompt: str, responses: Dict[str, str]) -> List[str]:
        """
        Get ranking of models for a specific prompt.
        
        Args:
            prompt: The prompt text
            responses: Dictionary mapping model keys to responses
            
        Returns:
            List of model keys ordered from best to worst
        """
        if self.reward_model is None:
            raise ValueError("Model not initialized. Call initialize_model() first.")
        
        scores = {}
        
        for model_key, response in responses.items():
            # Format conversation
            conversation = self._format_conversation(prompt, response)
            
            # Get reward score
            score = self.reward_model.get_reward(conversation)
            scores[model_key] = score
        
        # Sort by score (higher is better)
        return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    
    def get_ranking_with_scores(self, prompt: str, responses: Dict[str, str]) -> List[Tuple[str, float]]:
        """
        Get ranking with scores for a specific prompt.
        
        Args:
            prompt: The prompt text
            responses: Dictionary mapping model keys to responses
            
        Returns:
            List of (model_key, score) tuples ordered from best to worst
        """
        if self.reward_model is None:
            raise ValueError("Model not initialized. Call initialize_model() first.")
        
        scores = {}
        
        for model_key, response in responses.items():
            # Format conversation
            conversation = self._format_conversation(prompt, response)
            
            # Get reward score
            score = self.reward_model.get_reward(conversation)
            scores[model_key] = score
        
        # Sort by score (higher is better)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    def _format_conversation(self, prompt: str, response: str) -> str:
        """Format prompt and response as a conversation."""
        if hasattr(self.tokenizer, 'apply_chat_template'):
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response}
            ]
            return self.tokenizer.apply_chat_template(messages, tokenize=False)
        else:
            return f"User: {prompt}\nAssistant: {response}"
    
    def save_model(self, persona: str):
        """Save the trained model."""
        if self.reward_model is None:
            raise ValueError("No model to save")
        
        save_path = get_cache_file_path(f"reward_model_{persona}.pt")
        torch.save({
            'model_state_dict': self.reward_model.state_dict(),
            'config': {
                'base_model_name': self.reward_model.base_model_name,
                'lambda_reg': self.lambda_reg
            }
        }, save_path)
        
        print(f"Model saved to {save_path}")
    
    def load_model(self, persona: str):
        """Load a trained model."""
        load_path = get_cache_file_path(f"reward_model_{persona}.pt")
        
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"No saved model found at {load_path}")
        
        checkpoint = torch.load(load_path)
        
        # Initialize model
        self.initialize_model(checkpoint['config']['base_model_name'])
        
        # Load state dict
        self.reward_model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"Model loaded from {load_path}")
    
    def save_training_history(self, persona: str):
        """Save training history."""
        history_path = get_cache_file_path(f"reward_training_history_{persona}.json")
        
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
    
    def load_training_history(self, persona: str):
        """Load training history."""
        history_path = get_cache_file_path(f"reward_training_history_{persona}.json")
        
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                self.training_history = json.load(f)


if __name__ == "__main__":
    # Test the reward model system
    reward_system = PersonalizedRewardSystem()
    
    # Initialize model
    reward_system.initialize_model()
    
    # Test conversation formatting
    prompt = "Write a Python function to calculate fibonacci numbers"
    response = "Here's a Python function to calculate fibonacci numbers:\n\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"
    
    conversation = reward_system._format_conversation(prompt, response)
    print("Formatted conversation:")
    print(conversation)
    
    # Test reward scoring
    score = reward_system.reward_model.get_reward(conversation)
    print(f"Reward score: {score}") 