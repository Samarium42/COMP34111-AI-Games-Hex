# Monte Carlo Tree Search (MCTS)
This document describes the Monte Carlo Tree Search algorithm used by the agent, including selection, expansion, evaluation, and backup, as well as key enhancements and optimisations that improve performance and playing strength compared to a naïve MCTS implementation.

## Core algorithm
The agent uses Monte Carlo Tree Search as its primary decision-making algorithm. MCTS incrementally builds a search tree by running repeated simulations from the current game state. Each simulation consists of four phases: selection, expansion, evaluation, and backup. Over time, the tree statistics converge towards stronger move choices, allowing the agent to select moves based on accumulated evidence rather than fixed heuristics.

## Selection and PUCT
During the selection phase, the search tree is traversed from the root to a leaf node using a PUCT-style scoring function. This function balances exploitation of actions with high estimated value and exploration of less-visited actions. The policy prior provided by the neural network biases early exploration towards promising moves, while visit counts ensure that all actions are explored sufficiently. This approach allows the agent to make effective use of neural guidance under limited simulation budgets.

## Expansion and Evaluation
When a leaf node is reached, it is expanded by generating child nodes corresponding to legal moves. The current board state is then evaluated using the neural network, which produces both a policy distribution over actions and a scalar value estimate of the position. The policy output is used to initialise prior probabilities for child nodes, while the value estimate is used during backup to update node statistics.

## GRAVE and AMAF Statistics
To improve early-game performance, the agent incorporates GRAVE (Generalised Rapid Action Value Estimation), an AMAF-based enhancement to MCTS. GRAVE allows value information from similar actions played later in simulations to be shared with earlier nodes in the tree. This provides a stronger learning signal when visit counts are low and helps guide the search more effectively in the opening phase of the game.

## Backup and Statistics
After evaluation, the simulation result is propagated back up the tree during the backup phase. Node visit counts and value estimates are updated along the selected path. Both standard MCTS statistics and GRAVE-enhanced statistics are maintained, allowing the selection policy to blend long-term value estimates with rapid early-game feedback.

## Terminal States
Efficient detection of terminal states is critical for search performance. The agent uses BFS-based win checks for both Red and Blue players to determine whether a position is terminal. These checks are optimised to minimise overhead during search and allow early termination of simulations when a win is detected.

## Optimisations implemented
Several optimisations were applied to improve MCTS efficiency. Memory usage was reduced through careful node design and reuse of data structures. Redundant computations were avoided by precomputing commonly used values where possible. These optimisations collectively increase the number of simulations that can be performed within a fixed time budget.

## Simulations and Move selection
The number of simulations per move is fixed and tuned empirically to balance playing strength and response time. After all simulations are completed, the agent selects the move corresponding to the child node with the highest visit count. This approach provides more stable and reliable decisions than selecting moves purely based on value estimates.
