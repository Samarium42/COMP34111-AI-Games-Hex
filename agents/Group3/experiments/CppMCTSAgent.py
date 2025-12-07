"""
CppMCTSAgent - Agent wrapper for C++ MCTS engine with Azalea network
"""

import torch
import numpy as np
from src.AgentBase import AgentBase
from src.Board import Board
from src.Move import Move
from src.Colour import Colour
from agents.Group3.azalea_net import load_hex11_pretrained


class CppMCTSAgent(AgentBase):
    """
    Agent that uses C++ MCTS engine with neural network evaluation.
    """
    
    def __init__(self, colour: Colour, 
                 sims: int = 500,
                 model_path: str = "models/hex11-20180712-3362.policy.pth",
                 board_size: int = 11):
        super().__init__(colour)
        self.sims = sims
        self.board_size = board_size
        
        # Load neural network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[CppMCTSAgent] Loading Azalea network on {self.device}, sims={sims}")
        self.net = load_hex11_pretrained(model_path, self.device, board_size=board_size)
        self.net.eval()
        
        # Import C++ engine (deferred to avoid issues if not compiled)
        from interface import CppMCTS
        self.cpp_mcts = CppMCTS(board_size=board_size, sims=sims)
        
    def _board_to_flat(self, board: Board) -> np.ndarray:
        """Convert Board object to flat numpy array for C++."""
        N = board.size
        flat = np.zeros(N * N, dtype=np.int32)
        
        for x in range(N):
            for y in range(N):
                c = board.tiles[x][y].colour
                if c == Colour.RED:
                    flat[x * N + y] = 1
                elif c == Colour.BLUE:
                    flat[x * N + y] = 2
                # else: stays 0 (empty)
        
        return flat
    
    def _flat_to_tensor(self, flat_board: np.ndarray, player: int) -> torch.Tensor:
        """Convert flat board array to tensor for neural network."""
        N = self.board_size
        board_2d = flat_board.reshape(N, N)
        return torch.tensor(board_2d, dtype=torch.long, device=self.device)
    
    @torch.no_grad()
    def _evaluate_position(self, flat_board: np.ndarray, player: int):
        """
        Evaluate a position with the neural network.
        
        Returns:
            priors: np.ndarray shape (N*N,) - policy logits
            value: float - position value from player's perspective
        """
        x = self._flat_to_tensor(flat_board, player).unsqueeze(0)  # (1, N, N)
        
        logits, value = self.net(x)
        
        # Get policy priors (probabilities)
        priors = torch.softmax(logits[0], dim=0).cpu().numpy()
        
        # Value is from perspective of player in the encoded board
        # The C++ engine expects value from leaf player's perspective
        value_scalar = float(value.item())
        
        return priors, value_scalar
    
    def make_move(self, turn: int, board: Board, opponent_move: Move | None) -> Move:
        """
        Make a move using C++ MCTS engine.
        """
        N = board.size
        
        # Convert board to flat array
        board_flat = self._board_to_flat(board)
        
        # Player encoding: 1 = RED, 2 = BLUE
        player_int = 1 if self.colour == Colour.RED else 2
        
        # Reset C++ tree with current position
        self.cpp_mcts.reset(board_flat, player_int)
        
        # Run simulations
        for _ in range(self.sims):
            # Get leaf node from C++ engine
            leaf_board, leaf_player, is_terminal = self.cpp_mcts.request_leaf()
            
            if is_terminal:
                # Terminal nodes are handled by C++ engine
                # We still need to call apply_eval but it will be ignored
                dummy_priors = np.ones(N * N, dtype=np.float64) / (N * N)
                self.cpp_mcts.apply_eval(dummy_priors, 0.0)
            else:
                # Evaluate with neural network
                priors, value = self._evaluate_position(leaf_board, leaf_player)
                
                # Send evaluation back to C++
                self.cpp_mcts.apply_eval(priors, value)
        
        # Get best action from C++
        action_idx = self.cpp_mcts.best_action()
        
        x = action_idx // N
        y = action_idx % N
        
        return Move(x, y)


def test_cpp_agent():
    """Quick test to ensure C++ agent works."""
    print("Testing CppMCTSAgent...")
    
    from src.Board import Board
    
    board = Board(11)
    agent = CppMCTSAgent(Colour.RED, sims=50)
    
    move = agent.make_move(1, board, None)
    print(f"Agent made move: {move}")
    print("Test passed!")


if __name__ == "__main__":
    test_cpp_agent()