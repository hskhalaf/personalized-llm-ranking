#!/usr/bin/env python3
"""
Personalization & Evaluation Script

This script evaluates the personalized preference models by:
1. Loading the pre-trained global model
2. For each eligible user:
   - Split their data 80% train / 20% test
   - Perform online personalization training
   - Evaluate on test set with multiple metrics
3. Compare all models (MoE, Single Vector, Bradley-Terry)
4. Output comprehensive results table

Evaluation Metrics:
- Pairwise Accuracy: Fraction of correctly predicted preferences
- Log-Likelihood: Average log probability of observed preferences  
- NDCG@5: Normalized Discounted Cumulative Gain at rank 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import json
import yaml
import pickle
from pathlib import Path
import logging
from tqdm import tqdm
import sys
from typing import Dict, List, Tuple, Any
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))
from models import MoEPreferenceModel, SingleVectorModel, BradleyTerryModel, create_random_user_params

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PersonalizationEvaluator:
    """Main class for personalization evaluation."""
    
    def __init__(self, config_path: str = "configs/experiment_config.yaml"):
        """Initialize evaluator with configuration."""
        self.config_path = config_path
        self.config = self._load_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Set random seeds
        torch.manual_seed(self.config['random_seed'])
        np.random.seed(self.config['random_seed'])
        
        # Create results directory
        self.results_dir = Path(self.config['results_dir'])
        self.results_dir.mkdir(exist_ok=True)
        
    def _load_config(self) -> Dict:
        """Load experiment configuration."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_data(self) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], List[str], Dict[str, int]]:
        """Load all necessary data for evaluation."""
        logger.info("Loading cached data...")
        
        # Load processed comparisons
        comparisons_df = pd.read_parquet(self.config['cache_dir'] + '/arena_processed.parquet')
        
        # Load prompt embeddings
        with open(self.config['cache_dir'] + '/prompt_embeddings.pkl', 'rb') as f:
            prompt_embeddings = pickle.load(f)
        
        # Load personalization users
        with open(self.config['cache_dir'] + '/personalization_users.json', 'r') as f:
            personalization_users = json.load(f)
        
        # Load model vocabulary
        with open(self.config['cache_dir'] + '/model_vocab.json', 'r') as f:
            model_vocab = json.load(f)
        
        logger.info(f"Loaded {len(comparisons_df)} comparisons, {len(personalization_users)} users")
        return comparisons_df, prompt_embeddings, personalization_users, model_vocab
    
    def load_global_model(self, model_vocab: Dict[str, int]) -> MoEPreferenceModel:
        """Load and freeze the pre-trained global model."""
        logger.info("Loading pre-trained global model...")
        
        checkpoint = torch.load(self.config['results_dir'] + '/global_model_best.pt', map_location=self.device)
        
        # Create model
        model = MoEPreferenceModel(**checkpoint['model_params'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
        
        logger.info("Global model loaded and frozen")
        return model
    
    def split_user_data(self, user_data: pd.DataFrame, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split user's data into train/test sets."""
        n_train = int(len(user_data) * train_ratio)
        
        # Shuffle and split
        shuffled_data = user_data.sample(frac=1, random_state=self.config['random_seed']).reset_index(drop=True)
        train_data = shuffled_data.iloc[:n_train]
        test_data = shuffled_data.iloc[n_train:]
        
        return train_data, test_data
    
    def online_personalization_training(self, model: MoEPreferenceModel, train_data: pd.DataFrame, 
                                      prompt_embeddings: Dict[str, np.ndarray]) -> torch.Tensor:
        """Perform online personalization training for a user."""
        # Initialize user personalization matrix
        A_u = torch.zeros(self.config['num_experts_K'], self.config['low_rank_r'], 
                         requires_grad=True, device=self.device)
        
        # Set up SGD optimizer for A_u only
        optimizer = torch.optim.SGD([A_u], lr=self.config['user_learning_rate'])
        
        # Online training: process each comparison one by one
        for _, row in train_data.iterrows():
            # Skip if we don't have prompt embedding
            if row['prompt_text'] not in prompt_embeddings:
                continue
                
            # Get data
            prompt_embedding = torch.tensor(prompt_embeddings[row['prompt_text']], 
                                          dtype=torch.float32, device=self.device)
            winner_id = torch.tensor(row['winner_model_id'], dtype=torch.long, device=self.device)
            loser_id = torch.tensor(row['loser_model_id'], dtype=torch.long, device=self.device)
            
            # Forward pass
            winner_score = model(prompt_embedding, winner_id, A_u)
            loser_score = model(prompt_embedding, loser_id, A_u)
            
            # Compute loss with L2 regularization
            preference_loss = -F.logsigmoid(winner_score - loser_score)
            l2_penalty = self.config['user_l2_regularization'] * torch.sum(A_u ** 2)
            total_loss = preference_loss + l2_penalty
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        
        return A_u.detach()
    
    def compute_metrics(self, model, test_data: pd.DataFrame, prompt_embeddings: Dict[str, np.ndarray], 
                       user_params: torch.Tensor, model_type: str) -> Dict[str, float]:
        """Compute evaluation metrics on test data."""
        scores_winner = []
        scores_loser = []
        valid_comparisons = []
        
        # Collect scores for all valid test comparisons
        for _, row in test_data.iterrows():
            if row['prompt_text'] not in prompt_embeddings:
                continue
                
            prompt_embedding = torch.tensor(prompt_embeddings[row['prompt_text']], 
                                          dtype=torch.float32, device=self.device)
            winner_id = torch.tensor(row['winner_model_id'], dtype=torch.long, device=self.device)
            loser_id = torch.tensor(row['loser_model_id'], dtype=torch.long, device=self.device)
            
            with torch.no_grad():
                if model_type == 'bradley_terry':
                    winner_score = model(prompt_embedding, winner_id, user_params)
                    loser_score = model(prompt_embedding, loser_id, user_params)
                else:
                    winner_score = model(prompt_embedding, winner_id, user_params)
                    loser_score = model(prompt_embedding, loser_id, user_params)
            
            scores_winner.append(winner_score.cpu().item())
            scores_loser.append(loser_score.cpu().item())
            valid_comparisons.append(row)
        
        if len(scores_winner) == 0:
            return {'pairwise_accuracy': 0.0, 'log_likelihood': float('-inf'), 'ndcg_at_5': 0.0}
        
        scores_winner = np.array(scores_winner)
        scores_loser = np.array(scores_loser)
        
        # 1. Pairwise Accuracy
        correct_preferences = np.sum(scores_winner > scores_loser)
        pairwise_accuracy = correct_preferences / len(scores_winner)
        
        # 2. Log-Likelihood
        score_diffs = scores_winner - scores_loser
        log_probs = -np.log(1 + np.exp(-score_diffs))  # log_sigmoid
        log_likelihood = np.mean(log_probs)
        
        # 3. NDCG@5 (simplified version)
        # For each prompt, rank models by their scores and compute NDCG
        ndcg_scores = []
        
        # Group by prompt to compute ranking-based metrics
        prompt_groups = {}
        for i, row in enumerate(valid_comparisons):
            prompt = row['prompt_text']
            if prompt not in prompt_groups:
                prompt_groups[prompt] = []
            prompt_groups[prompt].append({
                'winner_model': row['winner_model_id'],
                'loser_model': row['loser_model_id'],
                'winner_score': scores_winner[i],
                'loser_score': scores_loser[i]
            })
        
        # Compute NDCG for each prompt (simplified)
        for prompt, comparisons in prompt_groups.items():
            if len(comparisons) < 2:
                continue
                
            # Collect all models and their scores for this prompt
            model_scores = {}
            for comp in comparisons:
                model_scores[comp['winner_model']] = comp['winner_score']
                model_scores[comp['loser_model']] = comp['loser_score']
            
            if len(model_scores) < 2:
                continue
            
            # Sort models by predicted scores (descending)
            sorted_models = sorted(model_scores.keys(), key=lambda m: model_scores[m], reverse=True)
            
            # Compute ideal ranking based on true preferences
            relevance_scores = {}
            for comp in comparisons:
                winner, loser = comp['winner_model'], comp['loser_model']
                relevance_scores[winner] = relevance_scores.get(winner, 0) + 1
                relevance_scores[loser] = relevance_scores.get(loser, 0) - 1
            
            # Compute DCG@5
            dcg = 0.0
            k = min(5, len(sorted_models))
            for i in range(k):
                model_id = sorted_models[i]
                relevance = max(0, relevance_scores.get(model_id, 0))  # Only positive relevance
                dcg += relevance / np.log2(i + 2)
            
            # Compute IDCG@5
            ideal_relevances = sorted([max(0, r) for r in relevance_scores.values()], reverse=True)
            idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevances[:k]) if rel > 0)
            
            # NDCG
            if idcg > 0:
                ndcg_scores.append(dcg / idcg)
        
        ndcg_at_5 = np.mean(ndcg_scores) if ndcg_scores else 0.0
        
        return {
            'pairwise_accuracy': pairwise_accuracy,
            'log_likelihood': log_likelihood,
            'ndcg_at_5': ndcg_at_5
        }
    
    def evaluate_moe_personalization(self, comparisons_df: pd.DataFrame, prompt_embeddings: Dict[str, np.ndarray], 
                                    personalization_users: List[str], model_vocab: Dict[str, int]) -> Dict[str, List[float]]:
        """Evaluate MoE model with personalization."""
        logger.info("Evaluating MoE model with personalization...")
        
        # Load global model
        global_model = self.load_global_model(model_vocab)
        
        results = {'pairwise_accuracy': [], 'log_likelihood': [], 'ndcg_at_5': []}
        
        # Process each user
        for user_id in tqdm(personalization_users, desc="MoE Personalization"):
            # Get user's data
            user_data = comparisons_df[comparisons_df['user_id'] == user_id]
            
            if len(user_data) < 10:  # Skip users with too little data
                continue
            
            # Split user data
            train_data, test_data = self.split_user_data(user_data)
            
            if len(train_data) == 0 or len(test_data) == 0:
                continue
            
            # Perform online personalization training
            A_u = self.online_personalization_training(global_model, train_data, prompt_embeddings)
            
            # Evaluate on test set
            metrics = self.compute_metrics(global_model, test_data, prompt_embeddings, A_u, 'moe')
            
            results['pairwise_accuracy'].append(metrics['pairwise_accuracy'])
            results['log_likelihood'].append(metrics['log_likelihood'])
            results['ndcg_at_5'].append(metrics['ndcg_at_5'])
        
        logger.info(f"MoE evaluation completed for {len(results['pairwise_accuracy'])} users")
        return results
    
    def evaluate_single_vector_personalization(self, comparisons_df: pd.DataFrame, prompt_embeddings: Dict[str, np.ndarray], 
                                             personalization_users: List[str], model_vocab: Dict[str, int]) -> Dict[str, List[float]]:
        """Evaluate Single Vector model with personalization."""
        logger.info("Evaluating Single Vector model with personalization...")
        
        # Create single vector model and copy global weights
        global_checkpoint = torch.load(self.config['results_dir'] + '/global_model_best.pt', map_location=self.device)
        single_model = SingleVectorModel(len(model_vocab), self.config['embedding_dim'], self.config['low_rank_r'])
        
        # Initialize with global model weights (adapting from K=2 to K=1)
        global_state = global_checkpoint['model_state_dict']
        single_state = single_model.state_dict()
        
        # Copy compatible parameters
        single_state['moe_model.v_m.weight'] = global_state['v_m.weight']
        single_state['moe_model.mu_G'] = global_state['mu_G'][:1]  # Take first expert only
        single_state['moe_model.B'] = global_state['B']
        
        single_model.load_state_dict(single_state)
        single_model = single_model.to(self.device)
        
        # Freeze global parameters
        for param in single_model.parameters():
            param.requires_grad = False
        
        results = {'pairwise_accuracy': [], 'log_likelihood': [], 'ndcg_at_5': []}
        
        # Process each user
        for user_id in tqdm(personalization_users, desc="Single Vector Personalization"):
            user_data = comparisons_df[comparisons_df['user_id'] == user_id]
            
            if len(user_data) < 10:
                continue
            
            train_data, test_data = self.split_user_data(user_data)
            
            if len(train_data) == 0 or len(test_data) == 0:
                continue
            
            # Initialize single user parameter (1 x r)
            A_u = torch.zeros(1, self.config['low_rank_r'], requires_grad=True, device=self.device)
            optimizer = torch.optim.SGD([A_u], lr=self.config['user_learning_rate'])
            
            # Online training
            for _, row in train_data.iterrows():
                if row['prompt_text'] not in prompt_embeddings:
                    continue
                    
                prompt_embedding = torch.tensor(prompt_embeddings[row['prompt_text']], 
                                              dtype=torch.float32, device=self.device)
                winner_id = torch.tensor(row['winner_model_id'], dtype=torch.long, device=self.device)
                loser_id = torch.tensor(row['loser_model_id'], dtype=torch.long, device=self.device)
                
                winner_score = single_model(prompt_embedding, winner_id, A_u)
                loser_score = single_model(prompt_embedding, loser_id, A_u)
                
                preference_loss = -F.logsigmoid(winner_score - loser_score)
                l2_penalty = self.config['user_l2_regularization'] * torch.sum(A_u ** 2)
                total_loss = preference_loss + l2_penalty
                
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
            
            # Evaluate
            A_u_final = A_u.detach()
            metrics = self.compute_metrics(single_model, test_data, prompt_embeddings, A_u_final, 'single')
            
            results['pairwise_accuracy'].append(metrics['pairwise_accuracy'])
            results['log_likelihood'].append(metrics['log_likelihood'])
            results['ndcg_at_5'].append(metrics['ndcg_at_5'])
        
        logger.info(f"Single Vector evaluation completed for {len(results['pairwise_accuracy'])} users")
        return results
    
    def evaluate_bradley_terry_personalization(self, comparisons_df: pd.DataFrame, prompt_embeddings: Dict[str, np.ndarray], 
                                             personalization_users: List[str], model_vocab: Dict[str, int]) -> Dict[str, List[float]]:
        """Evaluate Bradley-Terry model with personalization."""
        logger.info("Evaluating Bradley-Terry model with personalization...")
        
        # Create Bradley-Terry model
        bt_model = BradleyTerryModel(len(model_vocab), self.config['embedding_dim'])
        bt_model = bt_model.to(self.device)
        
        # Initialize with random weights and freeze
        for param in bt_model.parameters():
            param.requires_grad = False
        
        results = {'pairwise_accuracy': [], 'log_likelihood': [], 'ndcg_at_5': []}
        
        # Process each user
        for user_id in tqdm(personalization_users, desc="Bradley-Terry Personalization"):
            user_data = comparisons_df[comparisons_df['user_id'] == user_id]
            
            if len(user_data) < 10:
                continue
            
            train_data, test_data = self.split_user_data(user_data)
            
            if len(train_data) == 0 or len(test_data) == 0:
                continue
            
            # Initialize user preference vector
            w_u = torch.randn(self.config['embedding_dim'], requires_grad=True, device=self.device)
            optimizer = torch.optim.SGD([w_u], lr=self.config['user_learning_rate'])
            
            # Online training
            for _, row in train_data.iterrows():
                if row['prompt_text'] not in prompt_embeddings:
                    continue
                    
                prompt_embedding = torch.tensor(prompt_embeddings[row['prompt_text']], 
                                              dtype=torch.float32, device=self.device)
                winner_id = torch.tensor(row['winner_model_id'], dtype=torch.long, device=self.device)
                loser_id = torch.tensor(row['loser_model_id'], dtype=torch.long, device=self.device)
                
                winner_score = bt_model(prompt_embedding, winner_id, w_u)
                loser_score = bt_model(prompt_embedding, loser_id, w_u)
                
                preference_loss = -F.logsigmoid(winner_score - loser_score)
                l2_penalty = self.config['user_l2_regularization'] * torch.sum(w_u ** 2)
                total_loss = preference_loss + l2_penalty
                
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
            
            # Evaluate
            w_u_final = w_u.detach()
            metrics = self.compute_metrics(bt_model, test_data, prompt_embeddings, w_u_final, 'bradley_terry')
            
            results['pairwise_accuracy'].append(metrics['pairwise_accuracy'])
            results['log_likelihood'].append(metrics['log_likelihood'])
            results['ndcg_at_5'].append(metrics['ndcg_at_5'])
        
        logger.info(f"Bradley-Terry evaluation completed for {len(results['pairwise_accuracy'])} users")
        return results
    
    def aggregate_results(self, results: Dict[str, List[float]]) -> Dict[str, float]:
        """Aggregate results across users."""
        return {
            'pairwise_accuracy_mean': np.mean(results['pairwise_accuracy']),
            'pairwise_accuracy_std': np.std(results['pairwise_accuracy']),
            'log_likelihood_mean': np.mean(results['log_likelihood']),
            'log_likelihood_std': np.std(results['log_likelihood']),
            'ndcg_at_5_mean': np.mean(results['ndcg_at_5']),
            'ndcg_at_5_std': np.std(results['ndcg_at_5']),
            'num_users': len(results['pairwise_accuracy'])
        }
    
    def print_results_table(self, all_results: Dict[str, Dict[str, float]]):
        """Print formatted results table."""
        print("\n" + "="*80)
        print("PERSONALIZED LLM PREFERENCE MODEL EVALUATION RESULTS")
        print("="*80)
        
        # Create table
        models = list(all_results.keys())
        
        print(f"{'Model':<20} {'Accuracy':<15} {'Log-Likelihood':<15} {'NDCG@5':<15} {'Users':<10}")
        print("-" * 75)
        
        for model in models:
            results = all_results[model]
            print(f"{model:<20} "
                  f"{results['pairwise_accuracy_mean']:.4f}±{results['pairwise_accuracy_std']:.4f}   "
                  f"{results['log_likelihood_mean']:.4f}±{results['log_likelihood_std']:.4f}   "
                  f"{results['ndcg_at_5_mean']:.4f}±{results['ndcg_at_5_std']:.4f}   "
                  f"{results['num_users']:<10}")
        
        print("="*80)
        
        # Find best model for each metric
        best_accuracy = max(models, key=lambda m: all_results[m]['pairwise_accuracy_mean'])
        best_likelihood = max(models, key=lambda m: all_results[m]['log_likelihood_mean'])
        best_ndcg = max(models, key=lambda m: all_results[m]['ndcg_at_5_mean'])
        
        print(f"🏆 Best Accuracy: {best_accuracy}")
        print(f"🏆 Best Log-Likelihood: {best_likelihood}")  
        print(f"🏆 Best NDCG@5: {best_ndcg}")
        print("="*80)
    
    def run_evaluation(self) -> Dict[str, Any]:
        """Run complete evaluation pipeline."""
        logger.info("Starting personalization evaluation...")
        
        # Load data
        comparisons_df, prompt_embeddings, personalization_users, model_vocab = self.load_data()
        
        # Limit to subset for faster testing (remove in production)
        # personalization_users = personalization_users[:5]  # Uncomment for testing
        
        # Evaluate all models
        all_results = {}
        
        # 1. MoE Personalization
        moe_results = self.evaluate_moe_personalization(comparisons_df, prompt_embeddings, 
                                                       personalization_users, model_vocab)
        all_results['MoE + Personalization'] = self.aggregate_results(moe_results)
        
        # 2. Single Vector Personalization
        single_results = self.evaluate_single_vector_personalization(comparisons_df, prompt_embeddings, 
                                                                    personalization_users, model_vocab)
        all_results['Single Vector + Personalization'] = self.aggregate_results(single_results)
        
        # 3. Bradley-Terry Personalization
        bt_results = self.evaluate_bradley_terry_personalization(comparisons_df, prompt_embeddings, 
                                                                personalization_users, model_vocab)
        all_results['Bradley-Terry + Personalization'] = self.aggregate_results(bt_results)
        
        # Print results
        self.print_results_table(all_results)
        
        # Save results
        results_file = self.results_dir / f"personalization_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        logger.info(f"Results saved to {results_file}")
        
        return all_results

def main():
    """Main function to run personalization evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate personalized preference models")
    parser.add_argument("--config", default="configs/experiment_config.yaml", 
                       help="Path to configuration file")
    args = parser.parse_args()
    
    try:
        evaluator = PersonalizationEvaluator(args.config)
        results = evaluator.run_evaluation()
        print("\n✅ Personalization evaluation completed successfully!")
        return 0
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        print(f"\n❌ Personalization evaluation failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())