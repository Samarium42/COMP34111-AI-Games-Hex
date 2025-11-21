import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.AgentBase import AgentBase
from src.Board import Board
from src.Move import Move
from src.Colour import Colour


class HexState:
    def __init__(self, board: Board, player: Colour):
        self.board_size = board.size
        self.board = Board(board_size=self.board_size)
        for x in range(self.board_size):
            for y in range(self.board_size):
                self.board.tiles[x][y].colour = board.tiles[x][y].colour

        self.player = player  

    def clone(self):
        return HexState(self.board, self.player)

    def legal_moves(self):
        N = self.board_size
        moves = []
        for x in range(N):
            for y in range(N):
                if self.board.tiles[x][y].colour is None:
                    moves.append(x * N + y)
        return moves

    def play(self, action_index: int):
        new_state = self.clone()
        x = action_index // self.board_size
        y = action_index % self.board_size
        new_state.board.set_tile_colour(x, y, self.player)
        new_state.player = Colour.RED if self.player == Colour.BLUE else Colour.BLUE
        return new_state

    def is_terminal(self):
        red_win = self.board.has_ended(Colour.RED)
        blue_win = self.board.has_ended(Colour.BLUE)
        return red_win or blue_win

    def result(self):
        """
        Return value from the perspective of the player to move in this state.

        +1 : good for the current player
        -1 : bad for the current player
         0 : draw / no winner (should not really occur in Hex)
        """
        winner = self.board.get_winner()
        if winner is None:
            return 0
        if winner == self.player:
            return -1
        else:
            return 1

    def encode(self):
        N = self.board_size

        current = np.zeros((N, N), dtype=np.float32)
        opponent = np.zeros((N, N), dtype=np.float32)
        ones = np.ones((N, N), dtype=np.float32)

        for x in range(N):
            for y in range(N):
                c = self.board.tiles[x][y].colour
                if c == self.player:
                    current[x][y] = 1.0
                elif c is not None:
                    opponent[x][y] = 1.0

        x = torch.tensor(np.stack([current, opponent, ones], axis=0))
        return x

class ResNetBlock(nn.Module):
    def __init__(self, channels, reach=1, scale=1.0):
        super().__init__()
        self.conv = nn.Conv2d(
            channels, channels, kernel_size=2 * reach + 1, padding=reach, bias=False
        )
        self.bn = nn.BatchNorm2d(channels)
        self.scale = scale

    def forward(self, x):
        residual = x
        out = self.conv(x)
        out = self.bn(out)
        out = self.scale * out + residual
        return out * torch.sigmoid(out)


class HexResNet(nn.Module):
    def __init__(self, board_size=11, in_channels=3, channels=64, num_blocks=8):
        super().__init__()
        self.board_size = board_size

        self.conv_in = nn.Conv2d(
            in_channels, channels, kernel_size=3, padding=1, bias=False
        )
        self.bn_in = nn.BatchNorm2d(channels)

        self.trunk = nn.Sequential(
            *[ResNetBlock(channels, reach=1) for _ in range(num_blocks)]
        )


        self.policy_conv = nn.Conv2d(channels, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(
            2 * board_size * board_size, board_size * board_size
        )


        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        B = x.shape[0]
        out = F.relu(self.bn_in(self.conv_in(x)))
        out = self.trunk(out)

        p = F.relu(self.policy_bn(self.policy_conv(out)))
        p = p.view(B, -1)
        policy_logits = self.policy_fc(p)

        v = F.relu(self.value_bn(self.value_conv(out)))
        v = v.view(B, -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return policy_logits, v



class Node:
    def __init__(self, state: HexState, parent, prior):
        self.state = state
        self.parent = parent
        self.prior = float(prior)

        self.children: dict[int, "Node"] = {}
        self.N = 0
        self.W = 0.0
        self.Q = 0.0

    def expand(self, legal, priors):
        for a in legal:
            p = float(priors[a])
            self.children[a] = Node(self.state.play(a), self, p)

    def backup(self, value):
        node = self
        v = value
        while node is not None:
            node.N += 1
            node.W += v
            node.Q = node.W / node.N
            v = -v
            node = node.parent

    def select_child(self, c_puct):
        total_N = sum(child.N for child in self.children.values()) + 1e-8

        best_a = None
        best_child = None
        best_score = -1e9

        for a, child in self.children.items():
            U = c_puct * child.prior * (total_N ** 0.5) / (1 + child.N)
            score = child.Q + U
            if score > best_score:
                best_score = score
                best_a = a
                best_child = child

        return best_a, best_child


class MCTS:
    def __init__(self, net, sims=200, c_puct=1.2, device="cpu"):
        self.net = net
        self.sims = sims
        self.c_puct = c_puct
        self.device = device

    def run(self, root_state: HexState):
        root = Node(root_state, None, 1.0)
        self.expand_and_eval(root)

        for _ in range(self.sims):
            node = root


            while node.children and not node.state.is_terminal():
                _, node = node.select_child(self.c_puct)


            if node.state.is_terminal():
                value = float(node.state.result())
                node.backup(value)
                continue


            value = self.expand_and_eval(node)
            node.backup(value)

        N = root.state.board_size
        counts = np.zeros(N * N, dtype=np.float32)
        for a, child in root.children.items():
            counts[a] = child.N

        return counts

    @torch.no_grad()
    def expand_and_eval(self, node: Node):
        if node.state.is_terminal():
            return float(node.state.result())

        x = node.state.encode().unsqueeze(0).to(self.device)
        logits, value = self.net(x)
        logits = logits[0]
        value = value.item()

        priors = torch.softmax(logits, dim=0).cpu().numpy()

        legal = node.state.legal_moves()
        mask = np.zeros_like(priors)
        mask[legal] = 1.0
        priors = priors * mask

        if priors.sum() <= 0:
            priors = mask / mask.sum()
        else:
            priors = priors / priors.sum()

        node.expand(legal, priors)
        return value




class GraveNN(AgentBase):
    def __init__(self, colour: Colour):
        super().__init__(colour)
        self.board_size = 11
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = HexResNet(board_size=self.board_size).to(self.device)
        self.net.eval()
        self.mcts = MCTS(self.net, sims=200, device=self.device)

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        root_state = HexState(board, self.colour)

        counts = self.mcts.run(root_state)
        N = self.board_size

        best_action = int(np.argmax(counts))  

        x = best_action // N
        y = best_action % N
        return Move(x, y)
