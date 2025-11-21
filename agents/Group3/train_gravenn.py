import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from agents.Group3.GraveNN import GraveNN, HexState
from src.Board import Board
from src.Colour import Colour
from src.Move import Move

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Training hyperparameters
# ---------------------------
num_episodes = 10            # How many self-play games
mcts_sims = 200              # MCTS simulations per move
learning_rate = 1e-3
epochs = 3                   # Training epochs per batch

# Experience containers
states = []
policies = []
values = []


def choose_action(counts, temp=1.0):
    """Sample from visit counts."""
    counts = counts ** (1 / temp)
    probs = counts / counts.sum()
    return int(np.random.choice(len(probs), p=probs))


def play_self_play_game(net):
    """Generate a full game of self-play using MCTS."""
    from agents.Group3.GraveNN import MCTS

    N = 11
    board = Board(N)
    player = Colour.RED

    mcts = MCTS(net, sims=mcts_sims, device=device)
    game_states = []
    game_policies = []
    game_players = []

    while True:
        # Encode state
        hex_state = HexState(board, player)
        encoded = hex_state.encode().numpy()

        # Run MCTS
        counts = mcts.run(hex_state)

        # Record state and policy
        game_states.append(encoded)
        game_policies.append(counts)
        game_players.append(player)

        # Choose move
        action = choose_action(counts)
        x = action // N
        y = action % N

        board.set_tile_colour(x, y, player)

        # Check terminal
        if board.has_ended(player):
            winner = player
            break

        # Switch player
        player = Colour.RED if player == Colour.BLUE else Colour.BLUE

    # Assign final value labels (+1 win, -1 loss)
    for p in game_players:
        if p == winner:
            values.append(1.0)
        else:
            values.append(-1.0)

    # Store recorded states and policies
    states.extend(game_states)
    policies.extend(game_policies)


def train_network(net):
    """Train the network on accumulated experience."""
    if len(states) == 0:
        print("No experience collected yet")
        return

    print(f"Training on {len(states)} samples...")

    X = torch.tensor(np.array(states), dtype=torch.float32).to(device)
    P = torch.tensor(np.array(policies), dtype=torch.float32).to(device)
    V = torch.tensor(np.array(values), dtype=torch.float32).unsqueeze(1).to(device)

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        optimizer.zero_grad()

        logits, pred_values = net(X)

        # POLICY LOSS (cross entropy)
        log_probs = torch.log_softmax(logits, dim=1)
        policy_loss = -(P * log_probs).sum(dim=1).mean()

        # VALUE LOSS (MSE)
        value_loss = nn.MSELoss()(pred_values, V)

        loss = policy_loss + value_loss
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}: loss={loss.item():.4f}")


def main():
    net = GraveNN(Colour.RED).net  
    net.to(device)
    net.train()

    for episode in range(num_episodes):
        print(f"=== Self-play Game {episode+1}/{num_episodes} ===")
        play_self_play_game(net)
        train_network(net)

        torch.save(net.state_dict(), f"gravenn_checkpoint.pt")
        print("Saved checkpoint.")


if __name__ == "__main__":
    main()
