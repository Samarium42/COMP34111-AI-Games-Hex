# agents/Group3/train_selfplay.py

import random
import torch
import numpy as np
from collections import deque

from src.Colour import Colour
from src.Game import Game
from src.Player import Player

from agents.Group3.training_roster import ROSTER, make_learning_agent
from agents.Group3.noisy_agent import NoisyNumbaGraveNN
from agents.Group3.azalea_net import load_hex11_pretrained

# ======================================================
# CONFIG
# ======================================================
BOARD_SIZE = 11

# 100 games per iteration × 10 iterations = 1000 total games
GAMES_PER_ITER = 100
NUM_ITERS = 10

REPLAY_CAPACITY = 200_000
BATCH_SIZE = 256
LR = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
replay = deque(maxlen=REPLAY_CAPACITY)


# ======================================================
# Opponent sampling
# ======================================================
def sample_opponent() -> str:
    """
    0.5 = pure self-play
    0.4 = roster opponent
    0.1 = noisy self-play
    """
    r = random.random()
    if r < 0.5:
        return "self"
    elif r < 0.9:
        return random.choice(list(ROSTER.keys()))
    else:
        return "noisy_self"


def build_agents(ckpt_path: str) -> tuple[Player, Player]:
    """
    Build Player objects (with agents inside) for this game.
    Returns (red_player, blue_player).
    """
    choice = sample_opponent()

    if choice == "self":
        red_agent = make_learning_agent(Colour.RED, ckpt_path)
        blue_agent = make_learning_agent(Colour.BLUE, ckpt_path)

    elif choice == "noisy_self":
        red_agent = make_learning_agent(Colour.RED, ckpt_path)
        blue_agent = NoisyNumbaGraveNN(Colour.BLUE, load_path=ckpt_path)

    else:  # roster opponent name
        opp_name = choice
        if random.random() < 0.5:
            red_agent = make_learning_agent(Colour.RED, ckpt_path)
            blue_agent = ROSTER[opp_name](Colour.BLUE)
        else:
            red_agent = ROSTER[opp_name](Colour.RED)
            blue_agent = make_learning_agent(Colour.BLUE, ckpt_path)

    # Wrap into Player objects (this is what Game expects)
    red_player = Player("LearnerRed", red_agent)
    blue_player = Player("LearnerBlue", blue_agent)
    return red_player, blue_player


# ======================================================
# Single game with trajectory collection
# ======================================================
def run_single_game(red_player: Player, blue_player: Player):
    """
    Run one game and push (board, move_idx, z) for learning agent moves
    into the replay buffer.

    Assumes Game.run_with_trajectory() returns:
        winner_colour: Colour.RED / Colour.BLUE / None
        trajectory: list of dicts with:
            "board": (N,N) np.ndarray
            "player": Colour
            "move_idx": int
            "is_learning_agent": bool
    """
    game = Game(red_player, blue_player, board_size=BOARD_SIZE)

    winner_colour, trajectory = game.run_with_trajectory()

    for step in trajectory:
        board = step["board"]          # (N, N) ints 0/1/2
        player = step["player"]        # Colour
        move_idx = step["move_idx"]    # 0..N*N-1

        if winner_colour is None:
            z = 0.0
        elif player == winner_colour:
            z = 1.0
        else:
            z = -1.0

        replay.append((board.astype(np.int64), move_idx, z))


# ======================================================
# NN training step
# ======================================================
def train_one_epoch(net: torch.nn.Module, optimizer: torch.optim.Optimizer):
    if len(replay) < BATCH_SIZE:
        return

    net.train()
    for _ in range(200):  # 200 mini-batches per epoch
        batch = random.sample(replay, BATCH_SIZE)
        boards, moves, zs = zip(*batch)

        boards = torch.tensor(boards, dtype=torch.long, device=DEVICE)   # (B, N, N)
        moves = torch.tensor(moves, dtype=torch.long, device=DEVICE)     # (B,)
        zs = torch.tensor(zs, dtype=torch.float32, device=DEVICE)        # (B,)

        logits, values = net(boards)   # logits: (B, N*N), values: (B,1)
        values = values.squeeze(1)

        log_probs = torch.log_softmax(logits, dim=1)
        policy_loss = -log_probs[torch.arange(BATCH_SIZE, device=DEVICE), moves].mean()

        value_loss = torch.mean((values - zs) ** 2)

        loss = value_loss + policy_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# ======================================================
# Main training loop
# ======================================================
def main():
    ckpt_path = "models/hex11-20180712-3362.policy.pth"

    # Load Azalea net in our HexNetworkFull wrapper
    net = load_hex11_pretrained(ckpt_path, DEVICE)
    net.to(DEVICE)

    optimizer = torch.optim.Adam(net.parameters(), lr=LR)

    for it in range(NUM_ITERS):
        print(f"\n=== ITERATION {it+1}/{NUM_ITERS} ===")

        # 1) Generate self-play (and vs-roster) games
        for g in range(GAMES_PER_ITER):
            red_player, blue_player = build_agents(ckpt_path)
            run_single_game(red_player, blue_player)

        # 2) Train for one epoch on accumulated replay
        train_one_epoch(net, optimizer)

        # 3) Save updated checkpoint
        ckpt_path = f"models/selfplay_iter_{it+1}.pth"
        torch.save(net.state_dict(), ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
