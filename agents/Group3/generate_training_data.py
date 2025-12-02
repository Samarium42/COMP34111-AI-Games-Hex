"""
Generate training data from self-play games
Uses the MCTS agent to generate high-quality games
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.Board import Board
from src.Colour import Colour
from src.Move import Move
import importlib


def generate_self_play_game(game_id, time_per_move=0.3):
    """
    Generate one self-play game
    
    Args:
        game_id: Game number for logging
        time_per_move: Seconds per move (shorter = faster but weaker)
    
    Returns:
        List of training examples: [{board, move, color, result}, ...]
    """
    print(f"Starting game {game_id}...")
    
    # Import agents
    rey_module = importlib.import_module('agents.Group3.rey_agent')
    ReyAgent = rey_module.ReyAgent
    
    # Create board and agents
    board = Board(11)
    agent_red = ReyAgent(Colour.RED)
    agent_blue = ReyAgent(Colour.BLUE)
    
    game_history = []
    current_color = Colour.RED
    move_count = 0
    max_moves = 121  # 11x11 board
    
    while move_count < max_moves:
        agent = agent_red if current_color == Colour.RED else agent_blue
        
        # Get board state
        board_state = agent._board_to_numpy(board)
        
        # Get move (fast MCTS)
        try:
            move_coords = agent._parallel_mcts(board_state, time_limit=time_per_move)
        except:
            # Fallback to random legal move if MCTS fails
            from agents.Group3.rey_agent import get_legal_moves
            legal = get_legal_moves(board_state, 11)
            if len(legal) == 0:
                break
            move_coords = legal[np.random.randint(len(legal))]
        
        # Record position
        game_history.append({
            'board': board_state.copy(),
            'move': move_coords,
            'color': current_color.value + 1  # 1 for RED, 2 for BLUE
        })
        
        # Make move
        move = Move(move_coords[0], move_coords[1])
        board.set_tile_colour(move_coords[0], move_coords[1], current_color)
        
        # Check win
        if board.has_ended(current_color):
            winner = current_color.value + 1
            print(f"Game {game_id}: {current_color.name} wins after {move_count+1} moves")
            break
        
        # Switch player
        current_color = Colour.opposite(current_color)
        move_count += 1
    else:
        # Draw (shouldn't happen in Hex)
        winner = 0
        print(f"Game {game_id}: Draw after {move_count} moves")
    
    # Label all positions with outcome
    for entry in game_history:
        if winner == 0:
            entry['result'] = 0.0
        elif winner == entry['color']:
            entry['result'] = 1.0  # Win
        else:
            entry['result'] = -1.0  # Loss
    
    return game_history


def generate_training_dataset(num_games=50, save_path='training_data.npz'):
    """
    Generate multiple self-play games sequentially
    
    Args:
        num_games: Number of games to generate
        save_path: Where to save the data
    """
    print(f"\n{'='*60}")
    print(f"GENERATING {num_games} SELF-PLAY GAMES")
    print(f"{'='*60}\n")
    
    all_boards = []
    all_moves = []
    all_colors = []
    all_results = []
    
    for i in range(num_games):
        try:
            game_data = generate_self_play_game(i, time_per_move=0.3)
            
            for entry in game_data:
                all_boards.append(entry['board'])
                all_moves.append(entry['move'])
                all_colors.append(entry['color'])
                all_results.append(entry['result'])
            
            print(f"✓ Game {i+1}/{num_games} complete ({len(game_data)} positions)")
            
        except Exception as e:
            print(f"✗ Game {i+1}/{num_games} failed: {e}")
            continue
    
    # Convert to numpy arrays
    all_boards = np.array(all_boards, dtype=np.int8)
    all_moves = np.array(all_moves, dtype=np.int8)
    all_colors = np.array(all_colors, dtype=np.int8)
    all_results = np.array(all_results, dtype=np.float32)
    
    # Save
    np.savez_compressed(
        save_path,
        boards=all_boards,
        moves=all_moves,
        colors=all_colors,
        results=all_results
    )
    
    print(f"\n{'='*60}")
    print(f"DATASET GENERATED!")
    print(f"{'='*60}")
    print(f"Total positions: {len(all_boards)}")
    print(f"Saved to: {save_path}")
    print(f"File size: {os.path.getsize(save_path) / 1024 / 1024:.2f} MB")
    
    return all_boards, all_moves, all_colors, all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Hex training data')
    parser.add_argument('--games', type=int, default=50, help='Number of games to generate')
    parser.add_argument('--output', type=str, default='training_data.npz', help='Output file')
    
    args = parser.parse_args()
    
    generate_training_dataset(num_games=args.games, save_path=args.output)