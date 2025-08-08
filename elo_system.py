"""
Elo Offset System for personalized LLM ranking.
Modifies global ELO scores with user-specific offsets.
"""

import numpy as np
from typing import Dict, List, Tuple
import json
import os

from config import DEFAULT_MODELS, get_cache_file_path, CACHE_FILES


class EloOffsetSystem:
    def __init__(self, models: Dict = None, k_factor: float = 32.0):
        """
        Initialize the Elo Offset system.
        
        Args:
            models: Dictionary of models with their ELO scores
            k_factor: K-factor for Elo updates (default: 32.0)
        """
        self.models = models or DEFAULT_MODELS
        self.k_factor = k_factor
        
        # Initialize user-specific offsets
        self.user_offsets = {model_key: 0.0 for model_key in self.models.keys()}
        
        # Get base ELO scores
        self.base_elos = {}
        for model_key, model_info in self.models.items():
            if model_info['elo'] is not None:
                self.base_elos[model_key] = model_info['elo']
            else:
                print(f"Warning: No ELO score for {model_key}")
                self.base_elos[model_key] = 1200.0  # Default ELO
    
    def get_personalized_score(self, model_key: str) -> float:
        """
        Get the personalized score for a model.
        
        Args:
            model_key: Key of the model
            
        Returns:
            Personalized score = base_elo + user_offset
        """
        base_elo = self.base_elos.get(model_key, 1200.0)
        offset = self.user_offsets.get(model_key, 0.0)
        return base_elo + offset
    
    def get_all_personalized_scores(self) -> Dict[str, float]:
        """
        Get personalized scores for all models.
        
        Returns:
            Dictionary mapping model keys to personalized scores
        """
        return {model_key: self.get_personalized_score(model_key) 
                for model_key in self.models.keys()}
    
    def _expected_score(self, score_a: float, score_b: float) -> float:
        """
        Calculate expected score for player A against player B.
        
        Args:
            score_a: Score of player A
            score_b: Score of player B
            
        Returns:
            Expected score (probability of A winning)
        """
        return 1.0 / (1.0 + 10.0 ** ((score_b - score_a) / 400.0))
    
    def update_from_preference(self, winner: str, loser: str):
        """
        Update offsets based on a preference (winner > loser).
        
        Args:
            winner: Key of the winning model
            loser: Key of the losing model
        """
        # Get current personalized scores
        winner_score = self.get_personalized_score(winner)
        loser_score = self.get_personalized_score(loser)
        
        # Calculate expected scores
        expected_winner = self._expected_score(winner_score, loser_score)
        expected_loser = self._expected_score(loser_score, winner_score)
        
        # Calculate score changes
        winner_change = self.k_factor * (1.0 - expected_winner)
        loser_change = self.k_factor * (0.0 - expected_loser)
        
        # Update offsets
        self.user_offsets[winner] += winner_change
        self.user_offsets[loser] += loser_change
    
    def get_ranking(self) -> List[str]:
        """
        Get the current ranking of models based on personalized scores.
        
        Returns:
            List of model keys ordered from best to worst
        """
        scores = self.get_all_personalized_scores()
        return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    
    def get_ranking_with_scores(self) -> List[Tuple[str, float]]:
        """
        Get the current ranking with scores.
        
        Returns:
            List of (model_key, score) tuples ordered from best to worst
        """
        scores = self.get_all_personalized_scores()
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    def reset_offsets(self):
        """Reset all user offsets to zero."""
        self.user_offsets = {model_key: 0.0 for model_key in self.models.keys()}
    
    def save_state(self, persona: str):
        """Save the current state to cache."""
        cache_file = get_cache_file_path(f"elo_state_{persona}.json")
        
        state = {
            'user_offsets': self.user_offsets,
            'base_elos': self.base_elos,
            'k_factor': self.k_factor,
            'models': list(self.models.keys())
        }
        
        with open(cache_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, persona: str):
        """Load state from cache."""
        cache_file = get_cache_file_path(f"elo_state_{persona}.json")
        
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                state = json.load(f)
            
            self.user_offsets = state['user_offsets']
            self.base_elos = state['base_elos']
            self.k_factor = state.get('k_factor', 32.0)
    
    def get_model_info(self) -> Dict[str, Dict]:
        """
        Get comprehensive information about all models.
        
        Returns:
            Dictionary with model information including base ELO, offset, and personalized score
        """
        info = {}
        for model_key, model_info in self.models.items():
            base_elo = self.base_elos.get(model_key, 1200.0)
            offset = self.user_offsets.get(model_key, 0.0)
            personalized_score = self.get_personalized_score(model_key)
            
            info[model_key] = {
                'base_elo': base_elo,
                'user_offset': offset,
                'personalized_score': personalized_score,
                'arena_name': model_info.get('arena_name', 'Unknown'),
                'ollama_tag': model_info.get('ollama_tag', model_key)
            }
        
        return info
    
    def print_summary(self):
        """Print a summary of the current state."""
        print("\n=== Elo Offset System Summary ===")
        print(f"K-factor: {self.k_factor}")
        print("\nModel Rankings:")
        
        ranking = self.get_ranking_with_scores()
        for i, (model_key, score) in enumerate(ranking, 1):
            base_elo = self.base_elos.get(model_key, 1200.0)
            offset = self.user_offsets.get(model_key, 0.0)
            print(f"{i:2d}. {model_key:15s} | Base: {base_elo:6.1f} | Offset: {offset:+6.1f} | Total: {score:6.1f}")
        
        print()


if __name__ == "__main__":
    # Test the Elo Offset system
    elo_system = EloOffsetSystem()
    
    # Print initial state
    elo_system.print_summary()
    
    # Simulate some preferences
    print("Updating from preferences...")
    elo_system.update_from_preference('llama3.1:8b', 'mistral:7b')
    elo_system.update_from_preference('gemma3:4b', 'llama3.1:8b')
    elo_system.update_from_preference('qwen3:7b', 'gemma3:4b')
    
    # Print updated state
    elo_system.print_summary() 