"""
MCTS Simulation Count Optimization Experiment

Tests different simulation counts to find optimal balance between:
- Win rate (strength)
- Move speed (efficiency)

Metric: Wins per second vs simulation count
"""

import sys
import os
import time
import numpy as np
import json
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import List, Dict

# Add project root to path
sys.path.insert(0, '/mnt/project')

from src.Game import Game
from src.Player import Player
from src.Colour import Colour
from CppMCTSAgent import CppMCTSAgent


@dataclass
class GameResult:
    """Single game result with detailed metrics."""
    sim_count: int
    game_id: int
    winner: str  # "player1" or "player2"
    winner_colour: str  # "RED" or "BLUE"
    total_moves: int
    total_time: float
    player1_avg_time: float
    player2_avg_time: float
    player1_colour: str
    player2_colour: str


@dataclass
class ConfigStats:
    """Aggregate statistics for a simulation count configuration."""
    sim_count: int
    games_played: int
    red_wins: int
    blue_wins: int
    win_rate: float  # From RED's perspective
    avg_moves_per_game: float
    avg_time_per_move: float
    avg_game_time: float
    wins_per_second: float  # Key metric!
    
    
class MCTSExperiment:
    """
    Run MCTS simulation count experiments.
    """
    
    def __init__(self, 
                 sim_counts: List[int],
                 games_per_config: int = 50,
                 board_size: int = 11,
                 model_path: str = "models/hex11-20180712-3362.policy.pth"):
        
        self.sim_counts = sim_counts
        self.games_per_config = games_per_config
        self.board_size = board_size
        self.model_path = model_path
        
        self.results: List[GameResult] = []
        
    def run_single_game(self, sim_count: int, game_id: int, 
                       player1_colour: Colour) -> GameResult:
        """
        Run a single game with given simulation count.
        Alternate colors to control for first-move advantage.
        """
        print(f"\n{'='*60}")
        print(f"Game {game_id + 1}/{self.games_per_config} | Sims: {sim_count}")
        print(f"Player1 ({player1_colour.name}) vs Player2 ({Colour.opposite(player1_colour).name})")
        print(f"{'='*60}")
        
        # Create agents
        agent1 = CppMCTSAgent(player1_colour, sims=sim_count, 
                             model_path=self.model_path,
                             board_size=self.board_size)
        
        player2_colour = Colour.opposite(player1_colour)
        agent2 = CppMCTSAgent(player2_colour, sims=sim_count,
                             model_path=self.model_path, 
                             board_size=self.board_size)
        
        # Create players
        player1 = Player(name=f"Agent_S{sim_count}_P1", agent=agent1)
        player2 = Player(name=f"Agent_S{sim_count}_P2", agent=agent2)
        
        # Run game
        game = Game(player1, player2, 
                   board_size=self.board_size,
                   verbose=False, 
                   silent=True)
        
        start_time = time.time()
        result = game.run()
        end_time = time.time()
        
        # Parse results
        winner_name = result['winner']
        winner = "player1" if winner_name == player1.name else "player2"
        winner_colour = player1_colour.name if winner == "player1" else player2_colour.name
        
        total_moves = int(result['total_turns'])
        total_time = end_time - start_time
        
        # Average time per move
        p1_time = float(result['player_1_move_time'])
        p2_time = float(result['player_2_move_time'])
        p1_moves = int(result['player1_turns'])
        p2_moves = int(result['player2_turns'])
        
        p1_avg = p1_time / max(p1_moves, 1)
        p2_avg = p2_time / max(p2_moves, 1)
        
        game_result = GameResult(
            sim_count=sim_count,
            game_id=game_id,
            winner=winner,
            winner_colour=winner_colour,
            total_moves=total_moves,
            total_time=total_time,
            player1_avg_time=p1_avg,
            player2_avg_time=p2_avg,
            player1_colour=player1_colour.name,
            player2_colour=player2_colour.name
        )
        
        print(f"Winner: {winner_colour} ({winner})")
        print(f"Total moves: {total_moves}")
        print(f"Game time: {total_time:.2f}s")
        print(f"Avg move time: P1={p1_avg:.3f}s, P2={p2_avg:.3f}s")
        
        return game_result
    
    def run_configuration(self, sim_count: int) -> List[GameResult]:
        """
        Run all games for a specific simulation count.
        Alternate colors between games.
        """
        print(f"\n{'#'*70}")
        print(f"# TESTING CONFIGURATION: {sim_count} SIMULATIONS")
        print(f"# Games: {self.games_per_config}")
        print(f"{'#'*70}\n")
        
        config_results = []
        
        for game_id in range(self.games_per_config):
            # Alternate colors: even games RED first, odd games BLUE first
            player1_colour = Colour.RED if game_id % 2 == 0 else Colour.BLUE
            
            try:
                result = self.run_single_game(sim_count, game_id, player1_colour)
                config_results.append(result)
                self.results.append(result)
                
            except Exception as e:
                print(f"ERROR in game {game_id}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        return config_results
    
    def compute_statistics(self) -> Dict[int, ConfigStats]:
        """
        Compute aggregate statistics for each configuration.
        """
        stats_by_sim = defaultdict(lambda: {
            'games': 0,
            'red_wins': 0,
            'blue_wins': 0,
            'total_moves': 0,
            'total_game_time': 0,
            'total_move_time': 0,
            'total_moves_timed': 0
        })
        
        for result in self.results:
            s = stats_by_sim[result.sim_count]
            s['games'] += 1
            
            if result.winner_colour == 'RED':
                s['red_wins'] += 1
            else:
                s['blue_wins'] += 1
            
            s['total_moves'] += result.total_moves
            s['total_game_time'] += result.total_time
            
            # Track move times
            s['total_move_time'] += (result.player1_avg_time + result.player2_avg_time) / 2
            s['total_moves_timed'] += 1
        
        # Convert to ConfigStats objects
        config_stats = {}
        
        for sim_count, s in stats_by_sim.items():
            games = s['games']
            if games == 0:
                continue
            
            red_wins = s['red_wins']
            blue_wins = s['blue_wins']
            win_rate = red_wins / games  # From RED's perspective
            
            avg_moves = s['total_moves'] / games
            avg_game_time = s['total_game_time'] / games
            avg_move_time = s['total_move_time'] / max(s['total_moves_timed'], 1)
            
            # Key metric: wins per second
            # Approximate as: win_rate / avg_move_time
            # This represents how many wins you'd expect per second of computation
            wins_per_second = win_rate / avg_move_time if avg_move_time > 0 else 0
            
            config_stats[sim_count] = ConfigStats(
                sim_count=sim_count,
                games_played=games,
                red_wins=red_wins,
                blue_wins=blue_wins,
                win_rate=win_rate,
                avg_moves_per_game=avg_moves,
                avg_time_per_move=avg_move_time,
                avg_game_time=avg_game_time,
                wins_per_second=wins_per_second
            )
        
        return config_stats
    
    def run_all(self):
        """
        Run experiments for all simulation counts.
        """
        print("\n" + "="*70)
        print(" MCTS SIMULATION COUNT OPTIMIZATION EXPERIMENT")
        print("="*70)
        print(f"Simulation counts: {self.sim_counts}")
        print(f"Games per config: {self.games_per_config}")
        print(f"Board size: {self.board_size}")
        print(f"Total games: {len(self.sim_counts) * self.games_per_config}")
        print("="*70 + "\n")
        
        experiment_start = time.time()
        
        for sim_count in self.sim_counts:
            self.run_configuration(sim_count)
        
        experiment_end = time.time()
        total_time = experiment_end - experiment_start
        
        print("\n" + "="*70)
        print(" EXPERIMENT COMPLETE")
        print("="*70)
        print(f"Total time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
        print(f"Games completed: {len(self.results)}")
        print("="*70 + "\n")
        
        return self.compute_statistics()
    
    def save_results(self, output_dir: str = "/mnt/user-data/outputs"):
        """Save detailed results and statistics."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save raw results
        results_file = os.path.join(output_dir, "mcts_experiment_results.json")
        with open(results_file, 'w') as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)
        print(f"Saved raw results to: {results_file}")
        
        # Save statistics
        stats = self.compute_statistics()
        stats_file = os.path.join(output_dir, "mcts_experiment_stats.json")
        with open(stats_file, 'w') as f:
            json.dump({k: asdict(v) for k, v in stats.items()}, f, indent=2)
        print(f"Saved statistics to: {stats_file}")
        
        return stats


def print_summary_table(stats: Dict[int, ConfigStats]):
    """Print a nice summary table of results."""
    print("\n" + "="*100)
    print(" EXPERIMENT SUMMARY")
    print("="*100)
    
    print(f"{'Sims':<8} {'Games':<8} {'Win%':<8} {'AvgMoves':<10} "
          f"{'MoveTime(s)':<12} {'GameTime(s)':<12} {'Wins/Sec':<12}")
    print("-"*100)
    
    for sim_count in sorted(stats.keys()):
        s = stats[sim_count]
        print(f"{s.sim_count:<8} {s.games_played:<8} "
              f"{s.win_rate*100:<7.1f}% {s.avg_moves_per_game:<10.1f} "
              f"{s.avg_time_per_move:<12.3f} {s.avg_game_time:<12.1f} "
              f"{s.wins_per_second:<12.4f}")
    
    print("="*100 + "\n")


def main():
    """Run the complete experiment."""
    
    # Configuration
    SIM_COUNTS = [100, 500, 1000, 2000]
    GAMES_PER_CONFIG = 50
    BOARD_SIZE = 11
    MODEL_PATH = "/mnt/project/models/hex11-20180712-3362.policy.pth"
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print("Please ensure the Azalea model is available.")
        return
    
    # Create and run experiment
    experiment = MCTSExperiment(
        sim_counts=SIM_COUNTS,
        games_per_config=GAMES_PER_CONFIG,
        board_size=BOARD_SIZE,
        model_path=MODEL_PATH
    )
    
    stats = experiment.run_all()
    
    # Print summary
    print_summary_table(stats)
    
    # Save results
    experiment.save_results()
    
    print("\nExperiment complete! Results saved to /mnt/user-data/outputs/")


if __name__ == "__main__":
    main()