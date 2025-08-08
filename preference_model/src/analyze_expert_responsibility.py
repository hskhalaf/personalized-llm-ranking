#!/usr/bin/env python3
"""
Expert Responsibility Analysis

This script analyzes how the MoE model's experts specialize by:
1. Computing expert responsibility r_1(u,x,m_W) for each test example
2. Creating histograms to visualize specialization patterns
3. Performing qualitative analysis of expert-specific prompts

Key Questions:
- Are experts specializing (U-shaped histogram) or acting uniformly (peaked at 0.5)?
- What types of prompts does each expert handle?
- Is there interpretable specialization (e.g., coding vs creative writing)?
"""

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import json
import yaml
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from tqdm import tqdm
import sys
from typing import Dict, List, Tuple, Any
import argparse
from collections import defaultdict
import re

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))
from models import MoEPreferenceModel

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set style for better plots
try:
    plt.style.use('seaborn-v0_8')
except:
    try:
        plt.style.use('seaborn')
    except:
        plt.style.use('default')
sns.set_palette("husl")

class ExpertResponsibilityAnalyzer:
    """Analyzer for MoE expert responsibility and specialization."""
    
    def __init__(self, config_path: str = "configs/experiment_config.yaml"):
        """Initialize analyzer with configuration."""
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
        """Load all necessary data for analysis."""
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
        
        return comparisons_df, prompt_embeddings, personalization_users, model_vocab
    
    def load_global_model(self, model_vocab: Dict[str, int]) -> MoEPreferenceModel:
        """Load the pre-trained global model."""
        logger.info("Loading pre-trained global model...")
        
        checkpoint = torch.load(self.config['results_dir'] + '/global_model_best.pt', map_location=self.device)
        
        # Create model
        model = MoEPreferenceModel(**checkpoint['model_params'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def split_user_data(self, user_data: pd.DataFrame, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split user's data into train/test sets (same as evaluation script)."""
        n_train = int(len(user_data) * train_ratio)
        
        # Use same random seed for consistency
        shuffled_data = user_data.sample(frac=1, random_state=self.config['random_seed']).reset_index(drop=True)
        train_data = shuffled_data.iloc[:n_train]
        test_data = shuffled_data.iloc[n_train:]
        
        return train_data, test_data
    
    def train_user_personalization(self, model: MoEPreferenceModel, train_data: pd.DataFrame, 
                                  prompt_embeddings: Dict[str, np.ndarray]) -> torch.Tensor:
        """Train user personalization matrix (same as evaluation script)."""
        # Initialize user personalization matrix
        A_u = torch.zeros(self.config['num_experts_K'], self.config['low_rank_r'], 
                         requires_grad=True, device=self.device)
        
        # Set up SGD optimizer
        optimizer = torch.optim.SGD([A_u], lr=self.config['user_learning_rate'])
        
        # Online training
        for _, row in train_data.iterrows():
            if row['prompt_text'] not in prompt_embeddings:
                continue
                
            prompt_embedding = torch.tensor(prompt_embeddings[row['prompt_text']], 
                                          dtype=torch.float32, device=self.device)
            winner_id = torch.tensor(row['winner_model_id'], dtype=torch.long, device=self.device)
            loser_id = torch.tensor(row['loser_model_id'], dtype=torch.long, device=self.device)
            
            winner_score = model(prompt_embedding, winner_id, A_u)
            loser_score = model(prompt_embedding, loser_id, A_u)
            
            preference_loss = -F.logsigmoid(winner_score - loser_score)
            l2_penalty = self.config['user_l2_regularization'] * torch.sum(A_u ** 2)
            total_loss = preference_loss + l2_penalty
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        
        return A_u.detach()
    
    def compute_expert_responsibilities(self, model: MoEPreferenceModel, test_data: pd.DataFrame, 
                                     prompt_embeddings: Dict[str, np.ndarray], A_u: torch.Tensor) -> List[Dict]:
        """
        Compute expert responsibilities for each test example.
        
        For MoE model: r_1 = exp(s_1) / (exp(s_1) + exp(s_2))
        where s_k is the score from expert k before the log-sum-exp.
        """
        responsibilities = []
        
        with torch.no_grad():
            for _, row in test_data.iterrows():
                if row['prompt_text'] not in prompt_embeddings:
                    continue
                
                prompt_embedding = torch.tensor(prompt_embeddings[row['prompt_text']], 
                                              dtype=torch.float32, device=self.device)
                winner_id = torch.tensor(row['winner_model_id'], dtype=torch.long, device=self.device)
                
                # We need to compute the individual expert scores s_k before log-sum-exp
                # This requires modifying the forward pass to return intermediate values
                r_1_winner = self._compute_responsibility_for_model(model, prompt_embedding, winner_id, A_u)
                
                responsibilities.append({
                    'user_id': row['user_id'],
                    'prompt_text': row['prompt_text'],
                    'winner_model_id': row['winner_model_id'],
                    'winner_model_name': row['winner_model_name'],
                    'loser_model_id': row['loser_model_id'],
                    'loser_model_name': row['loser_model_name'],
                    'r_1_winner': r_1_winner,
                    'r_2_winner': 1 - r_1_winner  # Since r_1 + r_2 = 1
                })
        
        return responsibilities
    
    def _compute_responsibility_for_model(self, model: MoEPreferenceModel, prompt_embedding: torch.Tensor, 
                                        model_id: torch.Tensor, A_u: torch.Tensor) -> float:
        """
        Compute responsibility of expert 1 for a specific model.
        
        This requires manually computing the MoE forward pass to access intermediate values.
        """
        # Get model embedding
        v_m = model.v_m(model_id)  # (embedding_dim,)
        
        # Compute user-specific expert adjustments
        delta_u = torch.matmul(A_u, model.B.T)  # (num_experts_K, embedding_dim)
        
        # Compute expert scores s_k for each expert k
        expert_scores = []
        for k in range(model.num_experts_K):
            # Adjusted expert center
            adjusted_expert = model.mu_G[k] + delta_u[k]  # (embedding_dim,)
            
            # Score: (adjusted_expert * prompt_embedding).T @ v_m
            element_wise_product = adjusted_expert * prompt_embedding  # (embedding_dim,)
            s_k = torch.sum(element_wise_product * v_m)  # scalar
            expert_scores.append(s_k)
        
        # Convert to tensor
        expert_scores = torch.stack(expert_scores)  # (num_experts_K,)
        
        # Compute responsibilities using softmax
        responsibilities = F.softmax(expert_scores, dim=0)  # (num_experts_K,)
        
        return responsibilities[0].cpu().item()  # Return responsibility of expert 1
    
    def analyze_responsibility_distribution(self, all_responsibilities: List[Dict]) -> Dict[str, Any]:
        """Analyze the distribution of expert responsibilities."""
        r_1_values = [r['r_1_winner'] for r in all_responsibilities]
        
        # Basic statistics
        stats = {
            'mean': np.mean(r_1_values),
            'std': np.std(r_1_values),
            'median': np.median(r_1_values),
            'min': np.min(r_1_values),
            'max': np.max(r_1_values),
            'n_samples': len(r_1_values)
        }
        
        # Specialization analysis
        # Count how many are near 0.5 (uniform) vs near 0 or 1 (specialized)
        uniform_threshold = 0.1  # Within 0.1 of 0.5
        specialist_threshold = 0.1  # Within 0.1 of 0 or 1
        
        uniform_count = np.sum(np.abs(np.array(r_1_values) - 0.5) < uniform_threshold)
        expert1_specialist = np.sum(np.array(r_1_values) > (1 - specialist_threshold))  # r_1 > 0.9
        expert2_specialist = np.sum(np.array(r_1_values) < specialist_threshold)  # r_1 < 0.1
        
        specialization = {
            'uniform_count': uniform_count,
            'uniform_fraction': uniform_count / len(r_1_values),
            'expert1_specialist_count': expert1_specialist,
            'expert1_specialist_fraction': expert1_specialist / len(r_1_values),
            'expert2_specialist_count': expert2_specialist,
            'expert2_specialist_fraction': expert2_specialist / len(r_1_values),
            'total_specialist_fraction': (expert1_specialist + expert2_specialist) / len(r_1_values)
        }
        
        return {
            'statistics': stats,
            'specialization': specialization,
            'r_1_values': r_1_values
        }
    
    def create_responsibility_histogram(self, analysis_results: Dict[str, Any], save_path: str = None):
        """Create histogram of expert responsibilities."""
        r_1_values = analysis_results['r_1_values']
        stats = analysis_results['statistics']
        specialization = analysis_results['specialization']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Main histogram
        ax1.hist(r_1_values, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Uniform (0.5)')
        ax1.axvline(stats['mean'], color='green', linestyle='-', linewidth=2, label=f'Mean ({stats["mean"]:.3f})')
        
        ax1.set_xlabel('Expert 1 Responsibility (r₁)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Expert 1 Responsibilities\n(U-shaped = Good Specialization, Peak at 0.5 = No Specialization)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f"""Statistics:
Mean: {stats['mean']:.3f}
Std: {stats['std']:.3f}
Samples: {stats['n_samples']}

Specialization:
Expert 1 Dominant: {specialization['expert1_specialist_fraction']:.1%}
Expert 2 Dominant: {specialization['expert2_specialist_fraction']:.1%}
Uniform (≈0.5): {specialization['uniform_fraction']:.1%}
Total Specialist: {specialization['total_specialist_fraction']:.1%}"""
        
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Zoomed histogram focusing on tails and center
        ax2.hist(r_1_values, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
        ax2.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Uniform (0.5)')
        ax2.axvspan(0.4, 0.6, alpha=0.2, color='red', label='Uniform Region')
        ax2.axvspan(0, 0.1, alpha=0.2, color='blue', label='Expert 2 Dominant')
        ax2.axvspan(0.9, 1, alpha=0.2, color='green', label='Expert 1 Dominant')
        
        ax2.set_xlabel('Expert 1 Responsibility (r₁)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Expert Specialization Regions')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Histogram saved to {save_path}")
        
        plt.show()
    
    def analyze_prompt_specialization(self, all_responsibilities: List[Dict]) -> Dict[str, List[Dict]]:
        """Analyze what types of prompts each expert specializes in."""
        expert1_prompts = []  # r_1 > 0.9
        expert2_prompts = []  # r_1 < 0.1
        uniform_prompts = []  # 0.4 < r_1 < 0.6
        
        for resp in all_responsibilities:
            r_1 = resp['r_1_winner']
            
            if r_1 > 0.9:
                expert1_prompts.append(resp)
            elif r_1 < 0.1:
                expert2_prompts.append(resp)
            elif 0.4 < r_1 < 0.6:
                uniform_prompts.append(resp)
        
        return {
            'expert1_prompts': expert1_prompts,
            'expert2_prompts': expert2_prompts,
            'uniform_prompts': uniform_prompts
        }
    
    def categorize_prompts(self, prompts: List[str]) -> Dict[str, int]:
        """Categorize prompts by content type using simple keyword matching."""
        categories = {
            'coding': 0,
            'math': 0,
            'creative_writing': 0,
            'questions': 0,
            'instructions': 0,
            'comparison': 0,
            'explanation': 0,
            'other': 0
        }
        
        # Simple keyword-based categorization
        coding_keywords = ['code', 'program', 'function', 'python', 'javascript', 'html', 'css', 'sql', 'algorithm', 'debug', 'compile']
        math_keywords = ['calculate', 'equation', 'math', 'formula', 'solve', 'probability', 'statistics', 'geometry']
        creative_keywords = ['story', 'poem', 'creative', 'write', 'imagine', 'character', 'plot', 'fiction']
        question_keywords = ['what', 'how', 'why', 'when', 'where', 'who', 'which']
        instruction_keywords = ['create', 'make', 'build', 'generate', 'design', 'develop', 'implement']
        comparison_keywords = ['vs', 'versus', 'compare', 'difference', 'better', 'best', 'worst']
        explanation_keywords = ['explain', 'describe', 'tell me about', 'what is']
        
        for prompt in prompts:
            prompt_lower = prompt.lower()
            categorized = False
            
            if any(keyword in prompt_lower for keyword in coding_keywords):
                categories['coding'] += 1
                categorized = True
            elif any(keyword in prompt_lower for keyword in math_keywords):
                categories['math'] += 1
                categorized = True
            elif any(keyword in prompt_lower for keyword in creative_keywords):
                categories['creative_writing'] += 1
                categorized = True
            elif any(keyword in prompt_lower for keyword in comparison_keywords):
                categories['comparison'] += 1
                categorized = True
            elif any(keyword in prompt_lower for keyword in explanation_keywords):
                categories['explanation'] += 1 
                categorized = True
            elif any(keyword in prompt_lower for keyword in instruction_keywords):
                categories['instructions'] += 1
                categorized = True
            elif any(keyword in prompt_lower for keyword in question_keywords):
                categories['questions'] += 1
                categorized = True
            
            if not categorized:
                categories['other'] += 1
        
        return categories
    
    def print_qualitative_analysis(self, prompt_specialization: Dict[str, List[Dict]]):
        """Print qualitative analysis of expert specialization."""
        print("\n" + "="*80)
        print("EXPERT SPECIALIZATION ANALYSIS")
        print("="*80)
        
        for expert_name, prompts in prompt_specialization.items():
            if not prompts:
                continue
                
            print(f"\n📊 {expert_name.upper().replace('_', ' ')}: {len(prompts)} prompts")
            print("-" * 50)
            
            # Categorize prompts
            prompt_texts = [p['prompt_text'] for p in prompts]
            categories = self.categorize_prompts(prompt_texts)
            
            # Show category distribution
            total = len(prompts)
            print("Category Distribution:")
            for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
                if count > 0:
                    percentage = count / total * 100
                    print(f"  {category.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
            
            # Show sample prompts
            print(f"\nSample Prompts (showing up to 5):")
            for i, prompt_data in enumerate(prompts[:5]):
                prompt = prompt_data['prompt_text']
                r_1 = prompt_data['r_1_winner']
                # Truncate long prompts
                display_prompt = prompt[:100] + "..." if len(prompt) > 100 else prompt
                print(f"  {i+1}. (r₁={r_1:.3f}) {display_prompt}")
            
            if len(prompts) > 5:
                print(f"  ... and {len(prompts) - 5} more")
        
        print("\n" + "="*80)
    
    def run_analysis(self) -> Dict[str, Any]:
        """Run complete expert responsibility analysis."""
        logger.info("Starting expert responsibility analysis...")
        
        # Load data
        comparisons_df, prompt_embeddings, personalization_users, model_vocab = self.load_data()
        
        # Load model
        model = self.load_global_model(model_vocab)
        
        # Collect responsibilities across all users
        all_responsibilities = []
        
        logger.info("Computing expert responsibilities for each user...")
        for user_id in tqdm(personalization_users, desc="Analyzing users"):
            # Get user data
            user_data = comparisons_df[comparisons_df['user_id'] == user_id]
            
            if len(user_data) < 10:
                continue
            
            # Split data (same as evaluation)
            train_data, test_data = self.split_user_data(user_data)
            
            if len(test_data) == 0:
                continue
            
            # Train personalization
            A_u = self.train_user_personalization(model, train_data, prompt_embeddings)
            
            # Compute responsibilities for test data
            user_responsibilities = self.compute_expert_responsibilities(model, test_data, prompt_embeddings, A_u)
            all_responsibilities.extend(user_responsibilities)
        
        logger.info(f"Collected {len(all_responsibilities)} responsibility measurements")
        
        # Analyze distribution
        analysis_results = self.analyze_responsibility_distribution(all_responsibilities)
        
        # Create histogram
        hist_path = self.results_dir / "expert_responsibility_histogram.png"
        self.create_responsibility_histogram(analysis_results, str(hist_path))
        
        # Analyze prompt specialization
        prompt_specialization = self.analyze_prompt_specialization(all_responsibilities)
        
        # Print qualitative analysis
        self.print_qualitative_analysis(prompt_specialization)
        
        # Save detailed results (convert numpy types to Python types for JSON)
        def convert_numpy_types(obj):
            """Convert numpy types to Python types for JSON serialization."""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(v) for v in obj]
            return obj
        
        results = {
            'analysis_results': {
                'statistics': convert_numpy_types(analysis_results['statistics']),
                'specialization': convert_numpy_types(analysis_results['specialization'])
            },
            'prompt_specialization': {
                'expert1_count': len(prompt_specialization['expert1_prompts']),
                'expert2_count': len(prompt_specialization['expert2_prompts']),
                'uniform_count': len(prompt_specialization['uniform_prompts'])
            }
        }
        
        results_path = self.results_dir / "expert_responsibility_analysis.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Analysis results saved to {results_path}")
        
        return results

def main():
    """Main function to run expert responsibility analysis."""
    parser = argparse.ArgumentParser(description="Analyze MoE expert responsibility and specialization")
    parser.add_argument("--config", default="configs/experiment_config.yaml", 
                       help="Path to configuration file")
    args = parser.parse_args()
    
    try:
        analyzer = ExpertResponsibilityAnalyzer(args.config)
        results = analyzer.run_analysis()
        print("\n✅ Expert responsibility analysis completed successfully!")
        return 0
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        print(f"\n❌ Expert responsibility analysis failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())