# Hex AI Agent

## Overview
This project involves creating an AI agent to play the game Hex. The agent is based on MCTS combined with a neural network. A high performance C++ search engine is used with a Python-based neural network inference to balance search depth and evaluation quality.

## Key Features
The agent uses MCTS with PUCT-style selection to balance exploration and exploitation during search. Neural network policy and value heads are used to guide move selection and evaluate board states, reducin the reliance on random rollouts. GRAVE (AMAF-based) statistics are incorporated to improve early-game dcision making, when visit counts are lower. The game logic and core search are implemented in C++ for efficiency, and Python is used for neural network inference and training. Furthermore, batching neural network evaluation is used to reduce inference overhead.

## High-Level Architecture
The system consists of three main components. The C++ Hex engine implements the game logic, terminal state checks, and the MCTS search algorithm. A Python wrapper communicates with the C++ engine and handles neural network inference. The neural network module evaluates board states and returns policy priors and value estimates, which are integrated back into the C++ search tree. This separation allows the search to remain fast while still benefiting from learned evaluations.

## Agent functionality
During play, the C++ MCTS engine performs repeated simulations to explore possible future game states. When a leaf node is reached, the corresponding board state is passed to Python, where the neural network evaluates it and returns a policy distribution and value estimate. These results are used to expand the search tree and update node statistics. After a fixed number of simulations, the move with the highest visit count is selected.

## Structure
This repository is organised into source code, agent implementations, and documentation. Core game logic and search algorithms are implemented in C++. Python code is used for neural network inference, training, and interfacing with the C++ engine. The documentation folder contains detailed descriptions of the system architecture, MCTS implementation, and neural network design.

## Further Context
This project was developed as part of the coursework for the AI and Games module at the University of Manchester (COMP34111).
