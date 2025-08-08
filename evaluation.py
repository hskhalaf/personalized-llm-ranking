"""
Evaluation module for personalized LLM ranking experiment.
Handles metrics calculation and comparison between systems.
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from scipy.stats import kendalltau
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

from config import get_cache_file_path


class Evaluator:
    """Evaluator for comparing ranking systems."""
    
    def __init__(self):
        """Initialize the evaluator."""
        pass
    
    def kendall_tau(self, ranking1: List[str], ranking2: List[str]) -> float:
        """
        Calculate Kendall's Tau correlation between two rankings.
        
        Args:
            ranking1: First ranking (list of model keys)
            ranking2: Second ranking (list of model keys)
            
        Returns:
            Kendall's Tau correlation coefficient
        """
        # Create mapping from model to rank
        rank1 = {model: i for i, model in enumerate(ranking1)}
        rank2 = {model: i for i, model in enumerate(ranking2)}
        
        # Get common models
        common_models = set(ranking1) & set(ranking2)
        
        if len(common_models) < 2:
            return 0.0
        
        # Get ranks for common models
        ranks1 = [rank1[model] for model in common_models]
        ranks2 = [rank2[model] for model in common_models]
        
        # Calculate Kendall's Tau
        tau, p_value = kendalltau(ranks1, ranks2)
        
        return tau
    
    def spearman_rho(self, ranking1: List[str], ranking2: List[str]) -> float:
        """
        Calculate Spearman's Rho correlation between two rankings.
        
        Args:
            ranking1: First ranking (list of model keys)
            ranking2: Second ranking (list of model keys)
            
        Returns:
            Spearman's Rho correlation coefficient
        """
        from scipy.stats import spearmanr
        
        # Create mapping from model to rank
        rank1 = {model: i for i, model in enumerate(ranking1)}
        rank2 = {model: i for i, model in enumerate(ranking2)}
        
        # Get common models
        common_models = set(ranking1) & set(ranking2)
        
        if len(common_models) < 2:
            return 0.0
        
        # Get ranks for common models
        ranks1 = [rank1[model] for model in common_models]
        ranks2 = [rank2[model] for model in common_models]
        
        # Calculate Spearman's Rho
        rho, p_value = spearmanr(ranks1, ranks2)
        
        return rho
    
    def ranking_accuracy(self, predicted_ranking: List[str], ground_truth_ranking: List[str]) -> float:
        """
        Calculate ranking accuracy (fraction of correctly ordered pairs).
        
        Args:
            predicted_ranking: Predicted ranking
            ground_truth_ranking: Ground truth ranking
            
        Returns:
            Ranking accuracy (0.0 to 1.0)
        """
        # Create mapping from model to rank
        pred_rank = {model: i for i, model in enumerate(predicted_ranking)}
        gt_rank = {model: i for i, model in enumerate(ground_truth_ranking)}
        
        # Get common models
        common_models = set(predicted_ranking) & set(ground_truth_ranking)
        
        if len(common_models) < 2:
            return 0.0
        
        # Count correctly ordered pairs
        correct_pairs = 0
        total_pairs = 0
        
        for i, model1 in enumerate(common_models):
            for model2 in list(common_models)[i+1:]:
                # Check if the relative ordering is correct
                pred_order = pred_rank[model1] < pred_rank[model2]
                gt_order = gt_rank[model1] < gt_rank[model2]
                
                if pred_order == gt_order:
                    correct_pairs += 1
                
                total_pairs += 1
        
        return correct_pairs / total_pairs if total_pairs > 0 else 0.0
    
    def top_k_accuracy(self, predicted_ranking: List[str], ground_truth_ranking: List[str], k: int = 1) -> float:
        """
        Calculate top-k accuracy.
        
        Args:
            predicted_ranking: Predicted ranking
            ground_truth_ranking: Ground truth ranking
            k: Number of top models to consider
            
        Returns:
            Top-k accuracy
        """
        # Get top-k models
        pred_top_k = set(predicted_ranking[:k])
        gt_top_k = set(ground_truth_ranking[:k])
        
        # Calculate intersection
        intersection = pred_top_k & gt_top_k
        
        return len(intersection) / k
    
    def evaluate_system(self, system_rankings: Dict[str, List[str]], 
                       ground_truth_rankings: Dict[str, List[str]]) -> Dict[str, float]:
        """
        Evaluate a ranking system against ground truth.
        
        Args:
            system_rankings: Dictionary mapping prompt IDs to predicted rankings
            ground_truth_rankings: Dictionary mapping prompt IDs to ground truth rankings
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {
            'kendall_tau': [],
            'spearman_rho': [],
            'ranking_accuracy': [],
            'top_1_accuracy': [],
            'top_3_accuracy': []
        }
        
        # Calculate metrics for each prompt
        for prompt_id in ground_truth_rankings:
            if prompt_id in system_rankings:
                pred_ranking = system_rankings[prompt_id]
                gt_ranking = ground_truth_rankings[prompt_id]
                
                # Calculate metrics
                metrics['kendall_tau'].append(self.kendall_tau(pred_ranking, gt_ranking))
                metrics['spearman_rho'].append(self.spearman_rho(pred_ranking, gt_ranking))
                metrics['ranking_accuracy'].append(self.ranking_accuracy(pred_ranking, gt_ranking))
                metrics['top_1_accuracy'].append(self.top_k_accuracy(pred_ranking, gt_ranking, k=1))
                metrics['top_3_accuracy'].append(self.top_k_accuracy(pred_ranking, gt_ranking, k=3))
        
        # Calculate averages
        avg_metrics = {}
        for metric_name, values in metrics.items():
            if values:
                avg_metrics[metric_name] = np.mean(values)
                avg_metrics[f'{metric_name}_std'] = np.std(values)
            else:
                avg_metrics[metric_name] = 0.0
                avg_metrics[f'{metric_name}_std'] = 0.0
        
        return avg_metrics
    
    def compare_systems(self, elo_rankings: Dict[str, List[str]], 
                       reward_rankings: Dict[str, List[str]], 
                       ground_truth_rankings: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Compare the performance of both systems.
        
        Args:
            elo_rankings: Elo system rankings
            reward_rankings: Reward model system rankings
            ground_truth_rankings: Ground truth rankings
            
        Returns:
            Dictionary with comparison results
        """
        # Evaluate each system
        elo_metrics = self.evaluate_system(elo_rankings, ground_truth_rankings)
        reward_metrics = self.evaluate_system(reward_rankings, ground_truth_rankings)
        
        # Calculate differences
        differences = {}
        for metric in ['kendall_tau', 'spearman_rho', 'ranking_accuracy', 'top_1_accuracy', 'top_3_accuracy']:
            diff = reward_metrics[metric] - elo_metrics[metric]
            differences[f'{metric}_improvement'] = diff
        
        return {
            'elo_system': elo_metrics,
            'reward_system': reward_metrics,
            'differences': differences,
            'winner': 'reward' if differences['kendall_tau_improvement'] > 0 else 'elo'
        }
    
    def create_learning_curves(self, training_history: List[Dict], 
                              evaluation_results: List[Dict]) -> Dict[str, List[float]]:
        """
        Create learning curves from training and evaluation history.
        
        Args:
            training_history: List of training step results
            evaluation_results: List of evaluation results
            
        Returns:
            Dictionary with learning curve data
        """
        curves = {
            'steps': [],
            'elo_kendall_tau': [],
            'reward_kendall_tau': [],
            'elo_spearman_rho': [],
            'reward_spearman_rho': [],
            'elo_ranking_accuracy': [],
            'reward_ranking_accuracy': []
        }
        
        for i, eval_result in enumerate(evaluation_results):
            step = (i + 1) * 5  # Assuming evaluation every 5 steps
            
            curves['steps'].append(step)
            curves['elo_kendall_tau'].append(eval_result['elo_system']['kendall_tau'])
            curves['reward_kendall_tau'].append(eval_result['reward_system']['kendall_tau'])
            curves['elo_spearman_rho'].append(eval_result['elo_system']['spearman_rho'])
            curves['reward_spearman_rho'].append(eval_result['reward_system']['spearman_rho'])
            curves['elo_ranking_accuracy'].append(eval_result['elo_system']['ranking_accuracy'])
            curves['reward_ranking_accuracy'].append(eval_result['reward_system']['ranking_accuracy'])
        
        return curves
    
    def plot_learning_curves(self, curves: Dict[str, List[float]], 
                           save_path: str = None, persona: str = None):
        """
        Plot learning curves for both systems.
        
        Args:
            curves: Learning curve data
            save_path: Path to save the plot
            persona: Persona name for the plot title
        """
        plt.figure(figsize=(15, 5))
        
        # Kendall's Tau
        plt.subplot(1, 3, 1)
        plt.plot(curves['steps'], curves['elo_kendall_tau'], 'b-', label='Elo Offset', linewidth=2)
        plt.plot(curves['steps'], curves['reward_kendall_tau'], 'r-', label='Reward Model', linewidth=2)
        plt.xlabel('Training Steps')
        plt.ylabel("Kendall's Tau")
        plt.title("Kendall's Tau vs Training Steps")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Spearman's Rho
        plt.subplot(1, 3, 2)
        plt.plot(curves['steps'], curves['elo_spearman_rho'], 'b-', label='Elo Offset', linewidth=2)
        plt.plot(curves['steps'], curves['reward_spearman_rho'], 'r-', label='Reward Model', linewidth=2)
        plt.xlabel('Training Steps')
        plt.ylabel("Spearman's Rho")
        plt.title("Spearman's Rho vs Training Steps")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Ranking Accuracy
        plt.subplot(1, 3, 3)
        plt.plot(curves['steps'], curves['elo_ranking_accuracy'], 'b-', label='Elo Offset', linewidth=2)
        plt.plot(curves['steps'], curves['reward_ranking_accuracy'], 'r-', label='Reward Model', linewidth=2)
        plt.xlabel('Training Steps')
        plt.ylabel('Ranking Accuracy')
        plt.title('Ranking Accuracy vs Training Steps')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if persona:
            plt.suptitle(f'Learning Curves for {persona} Persona', fontsize=16, y=1.02)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Learning curves saved to {save_path}")
        
        plt.show()
    
    def plot_metric_comparison(self, comparison_results: Dict[str, Any], 
                             save_path: str = None, persona: str = None):
        """
        Plot comparison of final metrics between systems.
        
        Args:
            comparison_results: Results from compare_systems
            save_path: Path to save the plot
            persona: Persona name for the plot title
        """
        metrics = ['kendall_tau', 'spearman_rho', 'ranking_accuracy', 'top_1_accuracy', 'top_3_accuracy']
        
        elo_values = [comparison_results['elo_system'][metric] for metric in metrics]
        reward_values = [comparison_results['reward_system'][metric] for metric in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        plt.figure(figsize=(12, 6))
        bars1 = plt.bar(x - width/2, elo_values, width, label='Elo Offset', color='skyblue', alpha=0.8)
        bars2 = plt.bar(x + width/2, reward_values, width, label='Reward Model', color='lightcoral', alpha=0.8)
        
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title('Final Performance Comparison')
        plt.xticks(x, [metric.replace('_', ' ').title() for metric in metrics], rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        if persona:
            plt.suptitle(f'Performance Comparison for {persona} Persona', fontsize=16, y=1.02)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Metric comparison saved to {save_path}")
        
        plt.show()
    
    def save_evaluation_results(self, results: Dict[str, Any], persona: str):
        """Save evaluation results to file."""
        results_path = get_cache_file_path(f"evaluation_results_{persona}.json")
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Evaluation results saved to {results_path}")
    
    def load_evaluation_results(self, persona: str) -> Dict[str, Any]:
        """Load evaluation results from file."""
        results_path = get_cache_file_path(f"evaluation_results_{persona}.json")
        
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                return json.load(f)
        else:
            return None


if __name__ == "__main__":
    # Test the evaluator
    evaluator = Evaluator()
    
    # Test rankings
    ranking1 = ['model_a', 'model_b', 'model_c', 'model_d']
    ranking2 = ['model_b', 'model_a', 'model_d', 'model_c']
    
    tau = evaluator.kendall_tau(ranking1, ranking2)
    rho = evaluator.spearman_rho(ranking1, ranking2)
    accuracy = evaluator.ranking_accuracy(ranking1, ranking2)
    
    print(f"Kendall's Tau: {tau:.3f}")
    print(f"Spearman's Rho: {rho:.3f}")
    print(f"Ranking Accuracy: {accuracy:.3f}") 