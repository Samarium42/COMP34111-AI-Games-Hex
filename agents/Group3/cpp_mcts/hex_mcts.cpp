// agents/Group3/src/hex_mcts.cpp

#include <vector>
#include <random>
#include <cmath>
#include <limits>
#include <algorithm>
#include <cstring>
#include <ctime>
#include <cstdlib>
#include <iostream>

// Board encoding:
// 0 = empty
// 1 = RED
// 2 = BLUE
// Player 1 = RED connects left to right
// Player 2 = BLUE connects top to bottom

struct BoardState {
    int N;
    std::vector<int> cells; // size N*N

    BoardState(int n)
        : N(n), cells(n * n, 0) {}

    BoardState(int n, const int* flat)
        : N(n), cells(flat, flat + n * n) {}

    inline int& at(int x, int y) {
        return cells[x * N + y];
    }

    inline int at(int x, int y) const {
        return cells[x * N + y];
    }

    std::vector<int> legal_moves() const {
        std::vector<int> moves;
        moves.reserve(N * N);
        for (int i = 0; i < N * N; ++i) {
            if (cells[i] == 0) moves.push_back(i);
        }
        return moves;
    }

    void play(int action, int player) {
        cells[action] = player;
    }
};

static inline bool in_bounds(int N, int x, int y) {
    return (x >= 0 && x < N && y >= 0 && y < N);
}

static std::vector<std::pair<int,int>> hex_neighbors(int N, int x, int y) {
    static const int dx[6] = {-1, -1, 0, 0, 1, 1};
    static const int dy[6] = {0, 1, -1, 1, -1, 0};
    std::vector<std::pair<int,int>> out;
    out.reserve(6);
    for (int k = 0; k < 6; ++k) {
        int nx = x + dx[k];
        int ny = y + dy[k];
        if (in_bounds(N, nx, ny)) {
            out.emplace_back(nx, ny);
        }
    }
    return out;
}

// Return true if player has a connecting path
static bool has_winner(const BoardState& b, int player) {
    int N = b.N;
    std::vector<char> visited(N * N, 0);
    std::vector<std::pair<int,int>> stack;
    stack.reserve(N * N);

    if (player == 1) {
        // RED left to right
        for (int x = 0; x < N; ++x) {
            if (b.at(x, 0) == player) {
                stack.emplace_back(x, 0);
                visited[x * N + 0] = 1;
            }
        }
        while (!stack.empty()) {
            auto [x, y] = stack.back();
            stack.pop_back();
            if (y == N - 1) {
                return true;
            }
            auto nbrs = hex_neighbors(N, x, y);
            for (auto& p : nbrs) {
                int nx = p.first;
                int ny = p.second;
                int idx = nx * N + ny;
                if (!visited[idx] && b.at(nx, ny) == player) {
                    visited[idx] = 1;
                    stack.emplace_back(nx, ny);
                }
            }
        }
    } else {
        // BLUE top to bottom
        for (int y = 0; y < N; ++y) {
            if (b.at(0, y) == player) {
                stack.emplace_back(0, y);
                visited[0 * N + y] = 1;
            }
        }
        while (!stack.empty()) {
            auto [x, y] = stack.back();
            stack.pop_back();
            if (x == N - 1) {
                return true;
            }
            auto nbrs = hex_neighbors(N, x, y);
            for (auto& p : nbrs) {
                int nx = p.first;
                int ny = p.second;
                int idx = nx * N + ny;
                if (!visited[idx] && b.at(nx, ny) == player) {
                    visited[idx] = 1;
                    stack.emplace_back(nx, ny);
                }
            }
        }
    }
    return false;
}

// full terminal check
static int winner(const BoardState& b) {
    if (has_winner(b, 1)) return 1;
    if (has_winner(b, 2)) return 2;
    return 0;
}

// rollout from state where next player to move is "player"
static double random_rollout(BoardState state, int player, std::mt19937& rng) {
    int current = player;
    while (true) {
        int w = winner(state);
        if (w != 0) {
            if (w == player) return 1.0;
            else return -1.0;
        }
        auto moves = state.legal_moves();
        if (moves.empty()) {
            return 0.0;
        }
        std::uniform_int_distribution<int> dist(0, (int)moves.size() - 1);
        int action = moves[dist(rng)];
        state.play(action, current);
        current = (current == 1 ? 2 : 1);
    }
}

struct MCTSNode {
    BoardState state;
    int player_to_move;
    int last_action;
    MCTSNode* parent;
    std::vector<MCTSNode*> children;
    std::vector<int> unexpanded_actions;

    int N;
    int visits;
    double value_sum;

    MCTSNode(const BoardState& s, int player, MCTSNode* p, int action)
        : state(s),
          player_to_move(player),
          last_action(action),
          parent(p),
          N(s.N),
          visits(0),
          value_sum(0.0) 
    {
        unexpanded_actions = state.legal_moves();
    }

    bool is_fully_expanded() const {
        return unexpanded_actions.empty();
    }

    bool is_leaf() const {
        return children.empty();
    }

    double value() const {
        if (visits == 0) return 0.0;
        return value_sum / visits;
    }
};

// UCB child selection
static MCTSNode* select_child(MCTSNode* node, double c_puct) {
    double best_score = -1e18;
    MCTSNode* best_child = nullptr;

    for (MCTSNode* child : node->children) {
        double Q = child->value();
        double U = c_puct * std::sqrt((double)node->visits + 1e-8) / (1.0 + child->visits);
        double score = Q + U;
        if (score > best_score) {
            best_score = score;
            best_child = child;
        }
    }
    return best_child;
}

static MCTSNode* expand(MCTSNode* node, int player_to_move) {
    if (node->unexpanded_actions.empty()) return node;

    int action = node->unexpanded_actions.back();
    node->unexpanded_actions.pop_back();

    BoardState next_state = node->state;
    next_state.play(action, player_to_move);

    int next_player = (player_to_move == 1 ? 2 : 1);

    MCTSNode* child = new MCTSNode(next_state, next_player, node, action);
    node->children.push_back(child);
    return child;
}

static void backup(MCTSNode* node, double value, int perspective_player) {
    MCTSNode* cur = node;
    int current_player = node->player_to_move;
    while (cur != nullptr) {
        cur->visits += 1;
        double v = value;
        if (current_player != perspective_player) {
            v = -v;
        }
        cur->value_sum += v;

        current_player = (current_player == 1 ? 2 : 1);
        cur = cur->parent;
    }
}

static void delete_tree(MCTSNode* root) {
    if (!root) return;
    std::vector<MCTSNode*> stack;
    stack.push_back(root);
    while (!stack.empty()) {
        MCTSNode* node = stack.back();
        stack.pop_back();
        for (MCTSNode* c : node->children) {
            stack.push_back(c);
        }
        delete node;
    }
}

extern "C"
int hex_mcts_best_move(const int* board_flat,
                       int N,
                       int player,
                       int sims,
                       double c_puct)
{
    BoardState root_state(N, board_flat);
    MCTSNode* root = new MCTSNode(root_state, player, nullptr, -1);

    std::mt19937 rng((unsigned)std::time(nullptr));

    for (int i = 0; i < sims; ++i) {
        MCTSNode* node = root;

        while (!node->is_leaf() && node->is_fully_expanded()) {
            node = select_child(node, c_puct);
        }

        int w = winner(node->state);
        double value;

        if (w != 0) {
            value = (w == player ? 1.0 : -1.0);
        } else {
            if (!node->is_fully_expanded()) {
                node = expand(node, node->player_to_move);
            }
            BoardState rollout_state = node->state;
            int rollout_player = node->player_to_move;
            value = random_rollout(rollout_state, player, rng);
        }

        backup(node, value, player);
    }

    int best_action = -1;
    int best_visits = -1;

    for (MCTSNode* child : root->children) {
        if (child->visits > best_visits) {
            best_visits = child->visits;
            best_action = child->last_action;
        }
    }

    if (best_action < 0) {
        auto moves = root_state.legal_moves();
        if (!moves.empty()) best_action = moves[0];
    }

    delete_tree(root);
    return best_action;
}
