# agents/Group3/training_roster.py

from agents.Group3.NumbaGraveNN import NumbaGraveNN
from agents.TestAgents.DijkstraAgent import DijkstraAgent
from agents.TestAgents.MinimaxAgent import MinimaxAgent
from agents.TestAgents.PatternAgent import PatternAgent
from agents.TestAgents.HexAgent import HexAgent
from agents.TestAgents.GeneticAgent import GeneticAgent
from agents.TestAgents.DiyaAgent import DiyaAgent
from agents.TestAgents.MinimaxBeamAgent import MinimaxBeamAgent

from src.Colour import Colour

ROSTER = {
    "Dijkstra": lambda colour: DijkstraAgent(colour),
    "Minimax": lambda colour: MinimaxAgent(colour),
    "Pattern": lambda colour: PatternAgent(colour),
    "Hex": lambda colour: HexAgent(colour),
    "Genetic": lambda colour: GeneticAgent(colour),
    "Diya": lambda colour: DiyaAgent(colour),
    "MinimaxBeam": lambda colour: MinimaxBeamAgent(colour),
}

def make_learning_agent(colour, ckpt_path=None):
    return NumbaGraveNN(colour, load_path=ckpt_path or "models/hex11-20180712-3362.policy.pth")
