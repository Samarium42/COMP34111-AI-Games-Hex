from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move

import torch;
from torch import nn;

'''
For now just a basic neural network
- 11x11x2 input 
    - 11x11 board
        - (1,0) means red occupied square
        - (0,1) means blue occupied square
        - (0,0) means empty square


- 11x11 output of probabilities
    - Best move with highest probability
    - Should probably take best valid move for early models that output invalid moves


# High-level pseudocode for generational GA evolving NN policies

initialize population P of N individuals (random weights or some pretrained seeds)
HoF = []  # empty hall of fame

for gen in range(max_generations):
    # 1) Evaluate population with limited games
    for ind in P:
        opponents = sample_opponents(ind, P, HoF, fixed_baselines)
        ind.fitness = evaluate_winrate(ind, opponents, games_per_opponent)
    # 2) Elitism: keep top E
    elites = top_E(P)
    # 3) Selection + reproduction
    newP = elites.copy()
    while len(newP) < N:
        parentA = tournament_select(P)
        parentB = tournament_select(P)
        child = crossover(parentA, parentB)  # blend or uniform
        mutate(child)  # gaussian noise with adaptive sigma
        newP.append(child)
    P = newP
    # 4) Occasionally evaluate on longer budgets and update HoF
    if gen % hof_period == 0:
        best = top_1(P)
        if is_better_than_HoF(best, HoF):
            HoF.append(copy_of(best))  # keep a snapshot
            prune_HoF_if_needed(HoF)
    # 5) Logging / checkpoints / hyperparam adaptation
    log_metrics(P, HoF, gen)
    if stopping_criterion_met():
        break
        
    
    - Want to prioritize diversity as well, could end up with an orchestral approach
'''

from agents.TestAgents.utils import make_valid_move

class TestAgent(AgentBase):

     _board_size: int = 11

     def __init__(self, colour: Colour):
         super().__init__(colour)


     def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
         if turn == 2:
             return Move(-1, -1)
         else:
             return make_valid_move(board)
