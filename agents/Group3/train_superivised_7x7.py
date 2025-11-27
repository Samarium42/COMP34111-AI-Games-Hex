# agents/Group3/train_supervised_7x7.py

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from agents.Group3.GraveNN import HexResNet, HexState
from src.Board import Board
from src.Colour import Colour

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


class Hex7x7Dataset(Dataset):
    def __init__(self, boards, players_to_move, optimal_moves, values):
        """
        boards:          np.array (N, 7, 7) with {0, 1, 2}
        players_to_move: np.array (N,) with {1, 2} (map to Colour.RED/BLUE yourself)
        optimal_moves:   np.array (N,) int in [0, 48]
        values:          np.array (N,) float in [-1, 1]
        """
        self.boards = boards
        self.players_to_move = players_to_move
        self.optimal_moves = optimal_moves
        self.values = values

    def __len__(self):
        return len(self.boards)

    def __getitem__(self, idx):
        board_np = self.boards[idx]           # (7,7)
        who = self.players_to_move[idx]       # 1 or 2

        player = Colour.RED if who == 1 else Colour.BLUE

        # reuse your HexState.encode so encoding is consistent
        N = 7
        board = Board(N)
        # fill board tiles from board_np
        for x in range(N):
            for y in range(N):
                v = board_np[x, y]
                if v == 1:
                    board.tiles[x][y].colour = Colour.RED
                elif v == 2:
                    board.tiles[x][y].colour = Colour.BLUE

        state = HexState(board, player)
        x = state.encode(as_numpy=True)           # (4,7,7)

        pi_target = self.optimal_moves[idx]
        v_target = self.values[idx]

        return torch.tensor(x, dtype=torch.float32), int(pi_target), float(v_target)


def load_supervised_data(path="hex7x7_supervised.npz"):
    data = np.load(path)
    boards = data["boards"]              # (N,7,7)
    players_to_move = data["players_to_move"]  # (N,)
    optimal_moves = data["optimal_moves"]      # (N,)
    values = data["values"]                    # (N,)
    return boards, players_to_move, optimal_moves, values


def main():
    boards, players_to_move, optimal_moves, values = load_supervised_data()

    dataset = Hex7x7Dataset(boards, players_to_move, optimal_moves, values)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    model = HexResNet(board_size=7, in_channels=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    policy_loss_fn = nn.CrossEntropyLoss()
    value_loss_fn = nn.MSELoss()

    epochs = 20

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for x, pi_target, v_target in loader:
            x = x.to(device)                              # (B,4,7,7)
            pi_target = pi_target.to(device)              # (B,)
            v_target = v_target.to(device).unsqueeze(1)   # (B,1)

            optimizer.zero_grad()
            logits, v_pred = model(x)

            log_probs = torch.log_softmax(logits, dim=1)
            policy_loss = policy_loss_fn(log_probs, pi_target)
            value_loss = value_loss_fn(v_pred, v_target)

            loss = policy_loss + value_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}")

    torch.save(model.state_dict(), "hex7x7_supervised.pt")
    print("Saved 7x7 supervised weights to hex7x7_supervised.pt")


if __name__ == "__main__":
    main()
