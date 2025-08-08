"""
Main experiment orchestration for personalized LLM ranking.
Runs the complete experiment comparing Elo Offset vs Reward Model systems.
"""

import argparse
import json
import os
import time
from typing import Dict, List, Any

from config import PERSONAS, EXPERIMENT_CONFIG, get_cache_file_path
from data_generation import DataGenerator
from elo_system import EloOffsetSystem
from reward_model import PersonalizedRewardSystem
from evaluation import Evaluator


class PersonalizedRankingExperiment:
    """Main experiment class that orchestrates the entire experiment."""
    
    def __init__(self, persona: str = None, models: Dict = None, config: Dict = None):
        """
        Initialize the experiment.
        
        Args:
            persona: Persona to use for the experiment
            models: Custom models configuration
            config: Custom experiment configuration
        """
        self.persona = persona
        self.models = models
        self.config = config or EXPERIMENT_CONFIG
        
        # Initialize components
        self.data_generator = DataGenerator(models=self.models)
        self.elo_system = EloOffsetSystem(models=self.models)
        self.reward_system = PersonalizedRewardSystem(models=self.models, lambda_reg=self.config['lambda_reg'])
        self.evaluator = Evaluator()
        
        # Experiment state
        self.data = None
        self.evaluation_results = []
        self.final_results = None
    
    def run_data_generation(self) -> Dict[str, Any]:
        """Generate all data for the experiment."""
        print(f"\n{'='*60}")
        print(f"GENERATING DATA FOR PERSONA: {self.persona.upper()}")
        print(f"{'='*60}")
        
        # Generate all data
        self.data = self.data_generator.generate_all_data(self.persona)
        
        print(f"\nData Generation Complete:")
        print(f"- Prompts: {len(self.data['prompts'])}")
        print(f"- Preferences: {len(self.data['preferences'])}")
        print(f"- Test prompts: {len(self.data['test_prompts'])}")
        print(f"- Ground truth rankings: {len(self.data['ground_truth'])}")
        
        return self.data
    
    def run_training_and_evaluation(self) -> List[Dict[str, Any]]:
        """Run the training and evaluation loop."""
        print(f"\n{'='*60}")
        print(f"TRAINING AND EVALUATION")
        print(f"{'='*60}")
        
        if self.data is None:
            raise ValueError("No data available. Run run_data_generation() first.")
        
        preferences = self.data['preferences']
        responses = self.data['responses']
        test_prompts = self.data['test_prompts']
        ground_truth = self.data['ground_truth']
        
        # Initialize systems
        self.elo_system.reset_offsets()
        self.reward_system.initialize_model()
        
        # Training and evaluation loop
        evaluation_frequency = self.config['evaluation_frequency']
        evaluation_results = []
        
        print(f"Training on {len(preferences)} preferences...")
        print(f"Evaluating every {evaluation_frequency} preferences...")
        
        if self.config.get('incremental_training', False):
            # Incremental training approach (original)
            print("Using incremental training approach...")
            for i in range(0, len(preferences), evaluation_frequency):
                batch_end = min(i + evaluation_frequency, len(preferences))
                batch = preferences[i:batch_end]
                
                print(f"\nProcessing preferences {i+1}-{batch_end}...")
                
                # Update Elo system
                for pref in batch:
                    self.elo_system.update_from_preference(pref['winner'], pref['loser'])
                
                # Train reward model (if we have enough data)
                if i > 0:  # Skip first batch to have some training data
                    self.reward_system.train(
                        preferences[:batch_end], 
                        responses, 
                        max_epochs=1,  # Single epoch per batch
                        persona=self.persona
                    )
                
                # Evaluate both systems
                eval_result = self._evaluate_systems(test_prompts, responses, ground_truth)
                eval_result['preferences_processed'] = batch_end
                
                evaluation_results.append(eval_result)
                
                print(f"Evaluation {len(evaluation_results)}:")
                print(f"  Elo Kendall's Tau: {eval_result['elo_system']['kendall_tau']:.3f}")
                print(f"  Reward Kendall's Tau: {eval_result['reward_system']['kendall_tau']:.3f}")
        else:
            # Single training approach (simpler and faster)
            print("Using single training approach...")
            
            # Train reward model once on all training data
            print("\nTraining reward model on all training data...")
            self.reward_system.train(
                preferences, 
                responses, 
                max_epochs=self.config['max_epochs'],
                persona=self.persona
            )
            
            for i in range(0, len(preferences), evaluation_frequency):
                batch_end = min(i + evaluation_frequency, len(preferences))
                batch = preferences[i:batch_end]
                
                print(f"\nProcessing preferences {i+1}-{batch_end}...")
                
                # Update Elo system
                for pref in batch:
                    self.elo_system.update_from_preference(pref['winner'], pref['loser'])
                
                # Evaluate both systems
                eval_result = self._evaluate_systems(test_prompts, responses, ground_truth)
                eval_result['preferences_processed'] = batch_end
                
                evaluation_results.append(eval_result)
                
                print(f"Evaluation {len(evaluation_results)}:")
                print(f"  Elo Kendall's Tau: {eval_result['elo_system']['kendall_tau']:.3f}")
                print(f"  Reward Kendall's Tau: {eval_result['reward_system']['kendall_tau']:.3f}")
        
        self.evaluation_results = evaluation_results
        return evaluation_results
    
    def _evaluate_systems(self, test_prompts: List[str], responses: Dict, 
                         ground_truth: Dict) -> Dict[str, Any]:
        """Evaluate both systems on test prompts."""
        elo_rankings = {}
        reward_rankings = {}
        
        # Get Elo system ranking (same for all prompts)
        elo_ranking = self.elo_system.get_ranking()
        
        for prompt_id in test_prompts:
            if prompt_id in responses:
                prompt_responses = responses[prompt_id]
                
                # Elo system ranking (same for all prompts)
                elo_rankings[prompt_id] = elo_ranking
                
                # Reward model ranking (prompt-specific)
                try:
                    reward_ranking = self.reward_system.get_ranking(prompt_id, prompt_responses)
                    reward_rankings[prompt_id] = reward_ranking
                except Exception as e:
                    print(f"Warning: Failed to get reward ranking for {prompt_id}: {e}")
                    reward_rankings[prompt_id] = elo_ranking  # Fallback
        
        # Compare systems
        comparison = self.evaluator.compare_systems(
            elo_rankings, reward_rankings, ground_truth
        )
        
        return comparison
    
    def run_final_evaluation(self) -> Dict[str, Any]:
        """Run final comprehensive evaluation."""
        print(f"\n{'='*60}")
        print(f"FINAL EVALUATION")
        print(f"{'='*60}")
        
        if self.evaluation_results is None:
            raise ValueError("No evaluation results available. Run run_training_and_evaluation() first.")
        
        # Use the last evaluation result as final
        final_result = self.evaluation_results[-1]
        
        # Add learning curves
        curves = self.evaluator.create_learning_curves([], self.evaluation_results)
        final_result['learning_curves'] = curves
        
        # Add experiment metadata
        final_result['experiment_metadata'] = {
            'persona': self.persona,
            'num_prompts': len(self.data['prompts']),
            'num_preferences': len(self.data['preferences']),
            'num_test_prompts': len(self.data['test_prompts']),
            'lambda_reg': self.config['lambda_reg'],
            'models': list(self.models.keys()) if self.models else list(self.data_generator.models.keys())
        }
        
        self.final_results = final_result
        
        # Print final results
        self._print_final_results(final_result)
        
        return final_result
    
    def _print_final_results(self, results: Dict[str, Any]):
        """Print final experiment results."""
        print("\n" + "="*60)
        print("FINAL EXPERIMENT RESULTS")
        print("="*60)
        
        elo_metrics = results['elo_system']
        reward_metrics = results['reward_system']
        differences = results['differences']
        
        print(f"\nPersona: {self.persona}")
        print(f"Models: {', '.join(results['experiment_metadata']['models'])}")
        print(f"Lambda (regularization): {self.config['lambda_reg']}")
        
        print(f"\n{'Metric':<20} {'Elo Offset':<12} {'Reward Model':<12} {'Improvement':<12}")
        print("-" * 60)
        
        metrics = ['kendall_tau', 'spearman_rho', 'ranking_accuracy', 'top_1_accuracy', 'top_3_accuracy']
        
        for metric in metrics:
            elo_val = elo_metrics[metric]
            reward_val = reward_metrics[metric]
            improvement = differences[f'{metric}_improvement']
            
            print(f"{metric.replace('_', ' ').title():<20} {elo_val:<12.3f} {reward_val:<12.3f} {improvement:+<12.3f}")
        
        print(f"\nWinner: {results['winner'].upper()} SYSTEM")
        
        if differences['kendall_tau_improvement'] > 0:
            print("🎉 Reward Model system outperforms Elo Offset system!")
        else:
            print("📊 Elo Offset system performs better or equally.")
    
    def save_results(self):
        """Save all experiment results."""
        if self.final_results is None:
            raise ValueError("No final results to save. Run run_final_evaluation() first.")
        
        # Save final results
        self.evaluator.save_evaluation_results(self.final_results, self.persona)
        
        # Save learning curves plot
        curves = self.final_results['learning_curves']
        curves_path = get_cache_file_path(f"learning_curves_{self.persona}.png")
        self.evaluator.plot_learning_curves(curves, curves_path, self.persona)
        
        # Save metric comparison plot
        comparison_path = get_cache_file_path(f"metric_comparison_{self.persona}.png")
        self.evaluator.plot_metric_comparison(self.final_results, comparison_path, self.persona)
        
        # Save system states
        self.elo_system.save_state(self.persona)
        self.reward_system.save_model(self.persona)
        
        print(f"\nAll results saved for persona: {self.persona}")
    
    def run_complete_experiment(self) -> Dict[str, Any]:
        """Run the complete experiment from start to finish."""
        print(f"\n{'='*80}")
        print(f"STARTING PERSONALIZED LLM RANKING EXPERIMENT")
        print(f"Persona: {self.persona}")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        try:
            # Step 1: Generate data
            self.run_data_generation()
            
            # Step 2: Train and evaluate
            self.run_training_and_evaluation()
            
            # Step 3: Final evaluation
            self.run_final_evaluation()
            
            # Step 4: Save results
            self.save_results()
            
            elapsed_time = time.time() - start_time
            print(f"\nExperiment completed in {elapsed_time:.2f} seconds")
            
            return self.final_results
            
        except Exception as e:
            print(f"\nExperiment failed: {e}")
            raise


def main():
    """Main function to run the experiment."""
    parser = argparse.ArgumentParser(description="Personalized LLM Ranking Experiment")
    parser.add_argument("--persona", type=str, default="coder", 
                       choices=list(PERSONAS.keys()),
                       help="Persona to use for the experiment")
    parser.add_argument("--lambda_reg", type=float, default=EXPERIMENT_CONFIG['lambda_reg'],
                       help="Regularization strength for reward model")
    parser.add_argument("--num_prompts", type=int, default=EXPERIMENT_CONFIG['num_prompts'],
                       help="Number of prompts to generate")
    parser.add_argument("--eval_freq", type=int, default=EXPERIMENT_CONFIG['evaluation_frequency'],
                       help="Evaluation frequency")
    parser.add_argument("--incremental", action="store_true",
                       help="Use incremental training (slower but shows learning curves)")
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    config = EXPERIMENT_CONFIG.copy()
    config['lambda_reg'] = args.lambda_reg
    config['num_prompts'] = args.num_prompts
    config['evaluation_frequency'] = args.eval_freq
    config['incremental_training'] = args.incremental
    
    # Create and run experiment
    experiment = PersonalizedRankingExperiment(
        persona=args.persona,
        config=config
    )
    
    results = experiment.run_complete_experiment()
    
    print(f"\nExperiment completed successfully!")
    print(f"Results saved in cache directory")


if __name__ == "__main__":
    main() 