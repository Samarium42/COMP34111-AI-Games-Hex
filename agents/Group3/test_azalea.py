import torch
from agents.Group3.GraveNN import HexState
from src.Board import Board
from src.Colour import Colour
from agents.Group3.azalea_net import load_hex11_pretrained

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 1. Load Azalea model
net = load_hex11_pretrained(
    "models/hex11-20180712-3362.policy.pth",
    device,
    board_size=11
)

# 2. Create empty board + state
board = Board(11)
state = HexState(board, Colour.RED)

# 3. Encode input
x = state.encode(device=device).unsqueeze(0)  # (1, 11, 11)

# 4. Forward pass
logits, value = net(x)

print("Policy shape:", logits.shape)     # should be (1, 121)
print("Value shape:", value.shape)       # should be (1, 1)
print("Value:", value)

# 5. Try MCTS
from agents.Group3.GraveNN import MCTS
mcts = MCTS(net, sims=20, device=device)
counts = mcts.run(state)

print("MCTS counts shape:", counts.shape)  # (121,)
print("Counts sum:", counts.sum())
