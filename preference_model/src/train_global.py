#!/usr/bin/env python3
"""
Global Preference Model Training Script

This script trains the shared components (v_m, μ_G, B) of the MoEPreferenceModel
on the global dataset using pairwise preference comparisons.

The training process:
1. Load processed Arena data and prompt embeddings
2. Split data into 95% train / 5% validation
3. Train MoE model with zero user personalization (A_u = 0)
4. Use pairwise preference loss: -log_sigmoid(S_winner - S_loser)
5. Save trained model checkpoint
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pickle
import json
import yaml
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import sys
from typing import Dict, Tuple, List
import argparse
from datetime import datetime

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))
from models import MoEPreferenceModel

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PreferenceDataset(Dataset):
    """Dataset for pairwise preference comparisons."""
    
    def __init__(self, comparisons_df: pd.DataFrame, prompt_embeddings: Dict[str, np.ndarray]):
        """
        Initialize preference dataset.
        
        Args:
            comparisons_df: DataFrame with columns [prompt_text, winner_model_id, loser_model_id]
            prompt_embeddings: Dictionary mapping prompt text to embedding vectors
        """
        self.comparisons = comparisons_df.reset_index(drop=True)
        self.prompt_embeddings = prompt_embeddings
        
        # Filter out comparisons where we don't have prompt embeddings
        valid_indices = []
        for idx in range(len(self.comparisons)):
            prompt_text = self.comparisons.iloc[idx]['prompt_text']
            if prompt_text in self.prompt_embeddings:
                valid_indices.append(idx)
        
        self.comparisons = self.comparisons.iloc[valid_indices].reset_index(drop=True)
        logger.info(f"Dataset initialized with {len(self.comparisons)} valid comparisons")
        
    def __len__(self):
        return len(self.comparisons)
    
    def __getitem__(self, idx):
        """Get a single comparison example."""
        row = self.comparisons.iloc[idx]
        
        # Get prompt embedding
        prompt_text = row['prompt_text']
        prompt_embedding = torch.tensor(self.prompt_embeddings[prompt_text], dtype=torch.float32)
        
        # Get model IDs
        winner_id = torch.tensor(row['winner_model_id'], dtype=torch.long)
        loser_id = torch.tensor(row['loser_model_id'], dtype=torch.long)
        
        return {
            'prompt_embedding': prompt_embedding,
            'winner_id': winner_id,
            'loser_id': loser_id,
            'prompt_text': prompt_text  # For debugging
        }

class GlobalTrainer:
    """Trainer for global preference model."""
    
    def __init__(self, config_path: str = "configs/experiment_config.yaml"):
        """Initialize trainer with configuration."""
        self.config_path = config_path
        self.config = self._load_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Set random seeds for reproducibility
        torch.manual_seed(self.config['random_seed'])
        np.random.seed(self.config['random_seed'])
        
        # Create results directory
        self.results_dir = Path(self.config['results_dir'])
        self.results_dir.mkdir(exist_ok=True)
        
    def _load_config(self) -> Dict:
        """Load experiment configuration."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_data(self) -> Tuple[PreferenceDataset, PreferenceDataset]:
        """Load and split data into train/validation sets."""
        logger.info("Loading processed data...")
        
        # Load processed comparisons
        comparisons_df = pd.read_parquet(self.config['cache_dir'] + '/arena_processed.parquet')
        logger.info(f"Loaded {len(comparisons_df)} comparisons")
        
        # Load prompt embeddings
        with open(self.config['cache_dir'] + '/prompt_embeddings.pkl', 'rb') as f:
            prompt_embeddings = pickle.load(f)
        logger.info(f"Loaded {len(prompt_embeddings)} prompt embeddings")
        
        # Split data: 95% train, 5% validation
        train_size = int(0.95 * len(comparisons_df))
        
        # Shuffle the data
        shuffled_df = comparisons_df.sample(frac=1, random_state=self.config['random_seed']).reset_index(drop=True)
        
        train_df = shuffled_df.iloc[:train_size]
        val_df = shuffled_df.iloc[train_size:]
        
        logger.info(f"Split: {len(train_df)} train, {len(val_df)} validation")
        
        # Create datasets
        train_dataset = PreferenceDataset(train_df, prompt_embeddings)
        val_dataset = PreferenceDataset(val_df, prompt_embeddings)
        
        return train_dataset, val_dataset
    
    def create_model(self) -> MoEPreferenceModel:
        """Create and initialize the MoE preference model."""
        logger.info("Creating MoE preference model...")
        
        # Load model vocabulary to get number of models
        with open(self.config['cache_dir'] + '/model_vocab.json', 'r') as f:
            model_vocab = json.load(f)
        num_models = len(model_vocab)
        
        # Create model
        model = MoEPreferenceModel(
            num_models=num_models,
            embedding_dim=self.config['embedding_dim'],
            num_experts_K=self.config['num_experts_K'],
            low_rank_r=self.config['low_rank_r']
        )
        
        # Move to device
        model = model.to(self.device)
        
        logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
        return model
    
    def create_zero_user_params(self) -> torch.Tensor:
        """Create zero user personalization matrix A_u for global training."""
        A_u = torch.zeros(self.config['num_experts_K'], self.config['low_rank_r'])
        return A_u.to(self.device)
    
    def compute_loss(self, model: MoEPreferenceModel, batch: Dict, A_u: torch.Tensor) -> torch.Tensor:
        """Compute pairwise preference loss for a batch."""
        prompt_embeddings = batch['prompt_embedding'].to(self.device)
        winner_ids = batch['winner_id'].to(self.device)
        loser_ids = batch['loser_id'].to(self.device)
        
        # Compute scores for winners and losers
        winner_scores = model(prompt_embeddings, winner_ids, A_u)
        loser_scores = model(prompt_embeddings, loser_ids, A_u)
        
        # Pairwise preference loss: -log_sigmoid(S_winner - S_loser)
        score_diff = winner_scores - loser_scores
        loss = -F.logsigmoid(score_diff).mean()
        
        return loss, score_diff
    
    def evaluate(self, model: MoEPreferenceModel, dataloader: DataLoader, A_u: torch.Tensor) -> Dict:
        """Evaluate model on validation set."""
        model.eval()
        total_loss = 0.0
        correct_preferences = 0
        total_comparisons = 0
        
        with torch.no_grad():
            for batch in dataloader:
                loss, score_diff = self.compute_loss(model, batch, A_u)
                total_loss += loss.item()
                
                # Count correct preferences (winner should have higher score)
                correct_preferences += (score_diff > 0).sum().item()
                total_comparisons += len(score_diff)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_preferences / total_comparisons
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'correct': correct_preferences,
            'total': total_comparisons
        }
    
    def train(self) -> MoEPreferenceModel:
        """Train the global preference model."""
        logger.info("Starting global preference model training...")
        
        # Load data
        train_dataset, val_dataset = self.load_data()
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['global_batch_size'],
            shuffle=True,
            num_workers=0  # Set to 0 to avoid multiprocessing issues
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['global_batch_size'],
            shuffle=False,
            num_workers=0
        )
        
        # Create model
        model = self.create_model()
        
        # Create zero user personalization matrix
        A_u = self.create_zero_user_params()
        
        # Create optimizer (only optimize model parameters)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config['global_learning_rate'],
            weight_decay=self.config['global_weight_decay']
        )
        
        # Training loop
        best_val_accuracy = 0.0
        training_history = []
        
        for epoch in range(self.config['global_epochs']):
            logger.info(f"Epoch {epoch + 1}/{self.config['global_epochs']}")
            
            # Training phase
            model.train()
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            
            progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
            for batch_idx, batch in enumerate(progress_bar):
                optimizer.zero_grad()
                
                # Compute loss
                loss, score_diff = self.compute_loss(model, batch, A_u)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Statistics
                epoch_loss += loss.item()
                epoch_correct += (score_diff > 0).sum().item()
                epoch_total += len(score_diff)
                
                # Update progress bar
                current_accuracy = epoch_correct / epoch_total if epoch_total > 0 else 0.0
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{current_accuracy:.4f}'
                })
            
            # Calculate training metrics
            train_loss = epoch_loss / len(train_loader)
            train_accuracy = epoch_correct / epoch_total
            
            # Validation phase
            val_metrics = self.evaluate(model, val_loader, A_u)
            
            # Log results
            logger.info(
                f"Epoch {epoch + 1}: "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}"
            )
            
            # Save training history
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_accuracy': train_accuracy,
                'val_loss': val_metrics['loss'],
                'val_accuracy': val_metrics['accuracy']
            })
            
            # Save best model
            if val_metrics['accuracy'] > best_val_accuracy:
                best_val_accuracy = val_metrics['accuracy']
                self.save_model(model, f"global_model_best.pt")
                logger.info(f"New best validation accuracy: {best_val_accuracy:.4f}")
        
        # Save final model
        self.save_model(model, "global_model.pt")
        
        # Save training history
        history_path = self.results_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
        
        logger.info("Training completed!")
        logger.info(f"Best validation accuracy: {best_val_accuracy:.4f}")
        
        return model
    
    def save_model(self, model: MoEPreferenceModel, filename: str):
        """Save model checkpoint."""
        checkpoint_path = self.results_dir / filename
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'config': self.config,
            'model_params': {
                'num_models': model.num_models,
                'embedding_dim': model.embedding_dim,
                'num_experts_K': model.num_experts_K,
                'low_rank_r': model.low_rank_r
            },
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Model saved to {checkpoint_path}")

def main():
    """Main function to run global training."""
    parser = argparse.ArgumentParser(description="Train global preference model")
    parser.add_argument("--config", default="configs/experiment_config.yaml", 
                       help="Path to configuration file")
    args = parser.parse_args()
    
    try:
        trainer = GlobalTrainer(args.config)
        model = trainer.train()
        print("✅ Global training completed successfully!")
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        print(f"❌ Global training failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())