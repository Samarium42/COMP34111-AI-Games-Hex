import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

#torch.set_num_threads(1)

from agents.Group3.GraveNN import HexState, HexResNet
from src.Board import Board
from src.Colour import Colour
from src.Move import Move

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(">>> Training using MPS (Apple GPU)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(">>> Training using CUDA GPU")
else:
    device = torch.device("cpu")
    print(">>> Training using CPU (no GPU detected)")

# ---------------------------
# Training hyperparameters
# ---------------------------
num_episodes = 100          # How many self-play games
mcts_sims = 60              # MCTS simulations per move
learning_rate = 1e-3
epochs = 1               # Training epochs per batch

# Experience containers
states = []
policies = []
values = []


def choose_action(counts, temp=1.0):
    """Sample from visit counts."""
    counts = counts ** (1 / temp)
    probs = counts / counts.sum()
    return int(np.random.choice(len(probs), p=probs))


MAX_BUFFER = 2000  # or so

def add_experience(game_states, game_policies, game_values):
    global states, policies, values

    states.extend(game_states)
    policies.extend(game_policies)
    values.extend(game_values)

    # keep last MAX_BUFFER samples
    if len(states) > MAX_BUFFER:
        states = states[-MAX_BUFFER:]
        policies = policies[-MAX_BUFFER:]
        values = values[-MAX_BUFFER:]


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
        encoded = hex_state.encode(as_numpy=True)

        # Run MCTS
        counts = mcts.run(hex_state)
        pi = counts / counts.sum()

        # Record state and policy
        game_states.append(encoded)
        game_policies.append(pi)
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

    # Assign final value labels (+1 win, -1 loss) *locally*
    game_values = []
    for p in game_players:
        if p == winner:
            game_values.append(1.0)
        else:
            game_values.append(-1.0)

    return game_states, game_policies, game_values


def train_network(net, optimizer):
    """Train the network on accumulated experience."""
    if len(states) == 0:
        print("No experience collected yet")
        return

    n = min(len(states), len(policies), len(values))
    print(f"Training on {n} samples...")

    X = torch.tensor(np.array(states[:n]), dtype=torch.float32).to(device)
    P = torch.tensor(np.array(policies[:n]), dtype=torch.float32).to(device)
    V = torch.tensor(np.array(values[:n]), dtype=torch.float32).unsqueeze(1).to(device)

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
    net = HexResNet(board_size=11, in_channels=4).to(device)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    for episode in range(num_episodes):
        print(f"=== Self-play Game {episode+1}/{num_episodes} ===")

        # 1) generate one self-play game
        net.eval()
        game_states, game_policies, game_values = play_self_play_game(net)

        # 2) add to replay buffer (with MAX_BUFFER cap)
        add_experience(game_states, game_policies, game_values)
        print(f"Buffer sizes: states={len(states)}, policies={len(policies)}, values={len(values)}")

        # 3) train
        net.train()
        train_network(net, optimizer)

        # 4) save checkpoint
        torch.save(net.state_dict(), "gravenn_checkpoint.pt")
        print("Saved checkpoint.")


if __name__ == "__main__":
    main()
