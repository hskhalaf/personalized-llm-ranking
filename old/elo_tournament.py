import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import random

class Player:
    def __init__(self, player_id, true_skill, initial_elo=1500):
        self.id = player_id
        self.true_skill = true_skill
        self.elo = initial_elo
        self.games_played = 0

class EloTournament:
    def __init__(self, num_players=50, skill_mean=1500, skill_std=300, k_factor=32):
        self.k_factor = k_factor
        self.players = []
        
        # Generate players with normally distributed true skills
        true_skills = np.random.normal(skill_mean, skill_std, num_players)
        for i, skill in enumerate(true_skills):
            self.players.append(Player(i, skill))
    
    def expected_score(self, elo_a, elo_b):
        """Calculate expected score for player A against player B"""
        return 1 / (1 + 10**((elo_b - elo_a) / 400))
    
    def play_match(self, player_a, player_b):
        """Simulate a match between two players based on their true skills"""
        # Probability of player A winning based on true skill difference
        skill_diff = player_a.true_skill - player_b.true_skill
        prob_a_wins = 1 / (1 + 10**(-skill_diff / 400))
        
        # Simulate match outcome
        if random.random() < prob_a_wins:
            return 1, 0  # Player A wins
        else:
            return 0, 1  # Player B wins
    
    def update_elo(self, player_a, player_b, score_a, score_b):
        """Update Elo ratings after a match"""
        expected_a = self.expected_score(player_a.elo, player_b.elo)
        expected_b = 1 - expected_a
        
        # Update ratings
        player_a.elo += self.k_factor * (score_a - expected_a)
        player_b.elo += self.k_factor * (score_b - expected_b)
        
        player_a.games_played += 1
        player_b.games_played += 1
    
    def calculate_ranking_accuracy(self):
        """Calculate how well Elo rankings match true skill rankings"""
        # Sort players by true skill (descending)
        true_ranking = sorted(self.players, key=lambda p: p.true_skill, reverse=True)
        # Sort players by Elo rating (descending)
        elo_ranking = sorted(self.players, key=lambda p: p.elo, reverse=True)
        
        # Calculate Spearman rank correlation
        true_ranks = {player.id: i for i, player in enumerate(true_ranking)}
        elo_ranks = {player.id: i for i, player in enumerate(elo_ranking)}
        
        # Calculate correlation coefficient
        n = len(self.players)
        d_squared_sum = sum((true_ranks[pid] - elo_ranks[pid])**2 for pid in true_ranks)
        correlation = 1 - (6 * d_squared_sum) / (n * (n**2 - 1))
        
        return correlation
    
    def simulate_tournament(self, num_games=1000, sample_points=50):
        """Simulate tournament and track ranking accuracy"""
        accuracies = []
        game_counts = []
        
        games_per_sample = num_games // sample_points
        
        for game in range(num_games):
            # Select two random players
            player_a, player_b = random.sample(self.players, 2)
            
            # Play match and update ratings
            score_a, score_b = self.play_match(player_a, player_b)
            self.update_elo(player_a, player_b, score_a, score_b)
            
            # Sample accuracy at regular intervals
            if (game + 1) % games_per_sample == 0:
                accuracy = self.calculate_ranking_accuracy()
                accuracies.append(accuracy)
                game_counts.append(game + 1)
        
        return game_counts, accuracies

def run_simulation():
    """Run the tournament simulation and generate plot"""
    tournament = EloTournament(num_players=50, skill_mean=1500, skill_std=300)
    
    # Run simulation
    game_counts, accuracies = tournament.simulate_tournament(num_games=2000, sample_points=40)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(game_counts, accuracies, 'b-', linewidth=2, marker='o', markersize=4)
    plt.xlabel('Number of Games Played')
    plt.ylabel('Ranking Accuracy (Spearman Correlation)')
    plt.title('Elo Tournament: Ranking Accuracy vs Number of Games')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    # Add some statistics
    final_accuracy = accuracies[-1]
    plt.axhline(y=final_accuracy, color='r', linestyle='--', alpha=0.7, 
                label=f'Final Accuracy: {final_accuracy:.3f}')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print some results
    print(f"Tournament Results:")
    print(f"Players: {len(tournament.players)}")
    print(f"Total Games: {game_counts[-1]}")
    print(f"Final Ranking Accuracy: {final_accuracy:.3f}")
    
    # Show top 5 players by Elo vs true skill
    elo_sorted = sorted(tournament.players, key=lambda p: p.elo, reverse=True)
    skill_sorted = sorted(tournament.players, key=lambda p: p.true_skill, reverse=True)
    
    print(f"\nTop 5 by Elo Rating:")
    for i, player in enumerate(elo_sorted[:5]):
        print(f"{i+1}. Player {player.id}: Elo={player.elo:.1f}, True Skill={player.true_skill:.1f}")
    
    print(f"\nTop 5 by True Skill:")
    for i, player in enumerate(skill_sorted[:5]):
        print(f"{i+1}. Player {player.id}: True Skill={player.true_skill:.1f}, Elo={player.elo:.1f}")

if __name__ == "__main__":
    run_simulation()