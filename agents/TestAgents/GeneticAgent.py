import torch
import os

from src.Board import Board
from src.Colour import Colour
from src.Move import Move
from src.AgentBase import AgentBase
from agents.TestAgents.NeuralNetworkCUDA import NeuralNetworkCUDA

class GeneticAgent(AgentBase):
    def __init__(self, colour: Colour, model_path: str = None, network: NeuralNetworkCUDA = None):
        super().__init__(colour)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if network is not None:
            self.network = network
        else:
            if model_path is None:
                current_dir = os.path.dirname(os.path.abspath(__file__))
                model_path = os.path.join(current_dir, "best_brain.pt")

            if os.path.exists(model_path):
                try:
                    self.network = NeuralNetworkCUDA.load(model_path, device=device)
                    print(f"Loaded neural network from {model_path}")
                except Exception as e:
                    print(f"Failed to load {model_path}: {e}")
                    print("Using random network instead")
                    self.network = NeuralNetworkCUDA(device=device)
                    self.network.randomiseParameters()
            else:
                print(f"Warning: {model_path} not found, using random network")
                self.network = NeuralNetworkCUDA(device=device)
                self.network.randomiseParameters()

        self.network.eval() 

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        '''
        Make move using neural network
        '''

        is_player_two = self.colour == Colour.BLUE


        board_tensor = self.board_to_tensor(board)

        if is_player_two:
            board_tensor = board_tensor.permute(1, 0, 2)
            board_tensor = board_tensor[..., [1,0]]

        board_tensor = board_tensor.unsqueeze(0).to(self.network.device)

        with torch.no_grad():
            logits = self.network.forward(board_tensor).squeeze(0).cpu()
        
        legal_moves = self.get_legal_moves(board)
        legal_mask = torch.full (  (121,), float('-inf'))

        for move in legal_moves:

            if is_player_two:
                y_idx, x_idx = move.x, move.y # Indices are swapped
            else:
                y_idx, x_idx = move.y, move.x

            idx = y_idx * 11 + x_idx
            legal_mask[idx] = 0.0

        masked_logits = logits + legal_mask
        probabilities = torch.softmax(masked_logits, dim=0)

        move_idx = torch.argmax(probabilities).item()
        y_canonical = move_idx // 11
        x_canonical = move_idx % 11

        if is_player_two:
            # The network chose (x_canonical, y_canonical) on the transposed board.
            # This corresponds to (y_canonical, x_canonical) on the real board.
            final_x = y_canonical
            final_y = x_canonical
        else:
            final_x = x_canonical
            final_y = y_canonical

        return Move(final_x, final_y)
    def board_to_tensor(self, board: Board) -> torch.Tensor:
        '''
        Convert board to NN input format.
        '''

        tensor = torch.zeros(11, 11, 2)

        for y in range(11):
            for x in range(11):
                cell = board.tiles[x][y]
                if cell == self._colour:
                    tensor[y, x, 0] = 1.0
                elif cell == self.opp_colour():
                    tensor[y, x, 1] = 1.0
        return tensor
    
    def get_legal_moves(self, board: Board) -> list[Move]:
        '''
        Get all legal moves
        '''
        legal_moves = []
        for y in range(board.size):
            for x in range(board.size):
                if board.tiles[x][y].colour == None:
                    legal_moves.append(Move(x,y))
        return legal_moves