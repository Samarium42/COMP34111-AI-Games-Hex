#include <vector>
#include <queue>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <cstdint>
#include <random>
#include <unordered_map>


struct Node {
    std::vector<int> board;      // flat N*N board
    int player;                  // player to move (1=R, 2=B)
    int Nsize;                   // board size
    Node* parent;
    int action_from_parent;      // index 0..N*N-1
    uint64_t hash;               // Zobrist hash of this position


    // stats
    int visits;
    double value_sum;
    double Q;                    // value from this node's player's perspective

    // GRAVE/AMAF stats - track value of each action as if played first
    std::vector<int> amaf_visits;      // AMAF visit counts per action (size N*N)
    std::vector<double> amaf_value;    // AMAF value sum per action (size N*N)
    std::vector<double> Q_amaf;        // AMAF Q-value per action (size N*N)

    // children
    bool expanded;
    std::vector<Node*> children;
    std::vector<int> children_actions;   // action indices
    std::vector<double> priors;          // P(a|s)

    Node(const std::vector<int>& b, int p, int n, Node* par, int act)
        : board(b),
          player(p),
          Nsize(n),
          parent(par),
          action_from_parent(act),
          visits(0),
          value_sum(0.0),
          Q(0.0),
          expanded(false)
    {
        // Initialize GRAVE stats
        amaf_visits.resize(n * n, 0);
        amaf_value.resize(n * n, 0.0);
        Q_amaf.resize(n * n, 0.0);
    }
};


// =======================================================================
// Global root + settings
// =======================================================================

static Node* root = nullptr;
static Node* pending_leaf = nullptr;

static int BOARD_N = 11;
static double CP = 1.2;
static double REF = 0.5;  // GRAVE reference parameter (bias term)
static bool USE_GRAVE = true;  // Toggle GRAVE on/off

// =======================================================================
// Zobrist hashing
// =======================================================================

static std::vector<uint64_t> ZOBRIST;  // size = N*N*3 (0 empty, 1 red, 2 blue)
static bool ZOBRIST_INIT = false;

static void init_zobrist(int N) {
    // Re-init if board size changes or not yet initialised
    int needed = N * N * 3;
    if (ZOBRIST_INIT && (int)ZOBRIST.size() == needed) return;

    ZOBRIST.assign(needed, 0ULL);

    // Fixed seed => deterministic behaviour across runs (good for debugging)
    std::mt19937_64 rng(0xC0FFEE123456789ULL);

    for (int i = 0; i < needed; i++) {
        ZOBRIST[i] = rng();
    }

    ZOBRIST_INIT = true;
}

// index helper: (cell idx, piece {0,1,2}) -> zobrist entry
static inline uint64_t zob_key(int idx, int piece, int N) {
    return ZOBRIST[(idx * 3) + piece];
}

static uint64_t compute_hash(const std::vector<int>& b, int N) {
    uint64_t h = 0ULL;
    for (int i = 0; i < N * N; i++) {
        int piece = b[i]; // 0,1,2
        if (piece != 0) {
            h ^= zob_key(i, piece, N);
        }
    }
    return h;
}

// =======================================================================
// Terminal cache (keyed by Zobrist hash)
// =======================================================================
static std::unordered_map<uint64_t, uint8_t> WIN_CACHE;


// =======================================================================
// Hex win-checking utilities
// =======================================================================

bool red_wins(const std::vector<int>& b, int N) {
    // RED = 1 connects top to bottom (row 0 → row N-1)
    std::queue<int> q;
    std::vector<char> visited(N*N, 0);

    for (int y = 0; y < N; y++) {
        if (b[y] == 1) {
            q.push(y);
            visited[y] = 1;
        }
    }

    auto inside = [&](int x, int y){ return x>=0 && x<N && y>=0 && y<N; };
    int dx[6] = {-1,-1,0,1,1,0};
    int dy[6] = {0,1,1,0,-1,-1};

    while (!q.empty()) {
        int idx = q.front(); q.pop();
        int x = idx / N;
        int y = idx % N;

        if (x == N-1) return true;

        for (int k=0; k<6; k++) {
            int nx=x+dx[k], ny=y+dy[k];
            if (!inside(nx,ny)) continue;
            int id2 = nx*N + ny;
            if (!visited[id2] && b[id2] == 1) {
                visited[id2] = 1;
                q.push(id2);
            }
        }
    }
    return false;
}

bool blue_wins(const std::vector<int>& b, int N) {
    // BLUE = 2 connects left to right (col 0 → col N-1)
    std::queue<int> q;
    std::vector<char> visited(N*N, 0);

    for (int x=0; x<N; x++) {
        int idx = x*N + 0;
        if (b[idx] == 2) {
            q.push(idx);
            visited[idx] = 1;
        }
    }

    auto inside = [&](int x, int y){ return x>=0 && x<N && y>=0 && y<N; };
    int dx[6] = {-1,-1,0,1,1,0};
    int dy[6] = {0,1,1,0,-1,-1};

    while (!q.empty()) {
        int idx = q.front(); q.pop();
        int x = idx / N;
        int y = idx % N;

        if (y == N-1) return true;

        for (int k=0; k<6; k++) {
            int nx=x+dx[k], ny=y+dy[k];
            if (!inside(nx,ny)) continue;
            int id2 = nx*N + ny;
            if (!visited[id2] && b[id2] == 2) {
                visited[id2] = 1;
                q.push(id2);
            }
        }
    }
    return false;
}

static inline int cached_terminal_value(const Node* node) {
    // Look up cached winner for this position
    auto it = WIN_CACHE.find(node->hash);
    uint8_t winner;

    if (it != WIN_CACHE.end()) {
        winner = it->second;
    } else {
        // Compute winner once using BFS
        bool r  = red_wins(node->board, node->Nsize);
        bool bl = blue_wins(node->board, node->Nsize);

        if (!r && !bl) {
            winner = 0;
        } else if (r && !bl) {
            winner = 1;
        } else if (bl && !r) {
            winner = 2;
        } else {
            // Should not happen in Hex
            winner = 0;
        }

        WIN_CACHE[node->hash] = winner;
    }

    if (winner == 0) return 0;
    return (node->player == (int)winner) ? +1 : -1;
}


// =======================================================================
// Terminal check
// return:  +1 if 'player' (to move at this node) is the winner on this board
//          -1 if 'player' is the loser
//           0 if non-terminal
// =======================================================================

int check_terminal_value(const std::vector<int>& b, int player, int N) {
    bool r  = red_wins(b, N);
    bool bl = blue_wins(b, N);

    if (!r && !bl) {
        // no winner yet
        return 0;
    }

    int winner = 0;
    if (r && !bl) {
        winner = 1; // RED
    } else if (bl && !r) {
        winner = 2; // BLUE
    } else {
        // Should not happen in Hex (both cannot win).
        return 0;
    }

    return (player == winner) ? +1 : -1;
}


// =======================================================================
// GRAVE: Compute beta weighting for blending Q and Q_amaf
// Using the formula from Cazenave & Saffidine (2010):
// beta = sqrt(k / (3*n + k))
// where k is a constant (typically around 1000-5000)
// =======================================================================

double compute_beta(int n_visits, double k = 2000.0) {
    return std::sqrt(k / (3.0 * n_visits + k));
}


// =======================================================================
// Selection (PUCT with GRAVE)
// =======================================================================

Node* select_leaf(Node* node) {
    while (node->expanded && !node->children.empty()) {
        double totalN = 0.0;
        for (Node* c : node->children) totalN += c->visits;
        if (totalN < 1e-8) totalN = 1e-8;

        double best = -1e18;
        int best_i = 0;

        for (int i = 0; i < (int)node->children.size(); i++) {
            Node* c = node->children[i];
            int action = node->children_actions[i];

            // c->Q is from c->player's perspective.
            // For selection at 'node', we want value from node->player's perspective.
            double Q_mc = -c->Q;

            // Compute GRAVE blended Q-value
            double Q_combined = Q_mc;
            
            if (USE_GRAVE && c->visits > 0) {
                // Get AMAF statistics for this action
                double Q_amaf_val = 0.0;
                if (node->amaf_visits[action] > 0) {
                    Q_amaf_val = node->Q_amaf[action];
                }
                
                // Compute beta for blending
                double beta = compute_beta(c->visits);
                
                // GRAVE formula: Q_grave = (1-beta)*Q_mc + beta*(Q_amaf + ref)
                // ref is a bias term to encourage exploration
                Q_combined = (1.0 - beta) * Q_mc + beta * (Q_amaf_val + REF);
            }

            // PUCT exploration term
            double U = CP * node->priors[i] * std::sqrt(totalN) / (1.0 + c->visits);
            double score = Q_combined + U;
            
            if (score > best) {
                best = score;
                best_i = i;
            }
        }
        node = node->children[best_i];
    }
    return node;
}


// =======================================================================
// Expansion
// =======================================================================

void expand(Node* leaf, const double* priors) {
    if (leaf->expanded) return;

    int N = leaf->Nsize;
    std::vector<int> legal;

    for (int i=0; i<N*N; i++)
        if (leaf->board[i] == 0)
            legal.push_back(i);

    leaf->expanded = true;
    leaf->children_actions = legal;

    int L = (int)legal.size();
    leaf->priors.resize(L);
    leaf->children.resize(L, nullptr);

    for (int i=0; i<L; i++)
        leaf->priors[i] = priors[legal[i]];

    for (int i = 0; i < L; i++) {
        int a = legal[i];
        std::vector<int> nb = leaf->board;
        nb[a] = leaf->player;
        int nextp = (leaf->player == 1 ? 2 : 1);
        Node* child = new Node(nb, nextp, N, leaf, a);
        child->hash = leaf->hash ^ zob_key(a, leaf->player, N);
        leaf->children[i] = child;
}

}


// =======================================================================
// GRAVE Backup with AMAF updates
// Collect all actions played from leaf to root and update AMAF stats
// =======================================================================

void backup(Node* leaf, double value) {
    // Collect the sequence of actions from leaf back to root
    std::vector<int> action_sequence;
    std::vector<int> player_sequence;
    
    Node* cur = leaf;
    while (cur != nullptr && cur->parent != nullptr) {
        action_sequence.push_back(cur->action_from_parent);
        player_sequence.push_back(cur->parent->player);  // player who made this action
        cur = cur->parent;
    }
    
    // Reverse so we go root -> leaf
    std::reverse(action_sequence.begin(), action_sequence.end());
    std::reverse(player_sequence.begin(), player_sequence.end());

    // Now backup from leaf to root with AMAF updates
    cur = leaf;
    double v = value;

    while (cur != nullptr) {
        // Standard MCTS update
        cur->visits++;
        cur->value_sum += v;
        cur->Q = cur->value_sum / cur->visits;

        // GRAVE/AMAF update: update stats for all actions that could have been played
        // from this node, based on whether they appeared later in the playout
        if (USE_GRAVE && cur->parent != nullptr) {
            // Find position in sequence where we are
            int cur_depth = 0;
            Node* tmp = cur;
            while (tmp->parent != nullptr) {
                cur_depth++;
                tmp = tmp->parent;
            }
            
            // Update AMAF for actions that appear later in sequence
            for (size_t j = cur_depth; j < action_sequence.size(); j++) {
                int action = action_sequence[j];
                int action_player = player_sequence[j];
                
                // Only update if action was played by the same player as current node
                if (action_player == cur->player) {
                    cur->amaf_visits[action]++;
                    // The value from this action's perspective
                    // Since we track from cur's player perspective, and the action
                    // is played by the same player, use v directly
                    cur->amaf_value[action] += v;
                    cur->Q_amaf[action] = cur->amaf_value[action] / cur->amaf_visits[action];
                }
            }
        }

        v = -v;        // alternate perspective
        cur = cur->parent;
    }
}


// =======================================================================
// Root handling
// =======================================================================

void free_tree(Node* n) {
    if (!n) return;
    for (auto c: n->children) free_tree(c);
    delete n;
}

void init_root(const int* board, int N, int player) {
    BOARD_N = N;
    init_zobrist(N);

    std::vector<int> b(board, board + N*N);

    if (root) free_tree(root);
    root = new Node(b, player, N, nullptr, -1);
    root->hash = compute_hash(root->board, N);
    pending_leaf = nullptr;  // ensure no stale pointer between moves

    if (WIN_CACHE.size() > 200000) WIN_CACHE.clear();
}


// =======================================================================
// C API
// =======================================================================

extern "C" {

// Called at start of each Python make_move
void reset_tree(int* board, int N, int player) {
    init_root(board, N, player);
}

// Python asks for a leaf
void request_leaf(int* out_board, int* out_player, int* out_is_terminal) {
    if (!root) {
        *out_is_terminal = 1;
        return;
    }

    Node* leaf = select_leaf(root);
    pending_leaf = leaf;

    int term = cached_terminal_value(leaf);
    *out_is_terminal = (term != 0);


    int NN = leaf->Nsize * leaf->Nsize;
    for (int i=0; i<NN; i++)
        out_board[i] = leaf->board[i];
    *out_player = leaf->player;
}


// Python returns priors + value for pending_leaf
// 'value' is from leaf->player's perspective (Python already flips if needed).
void apply_eval(const double* priors, double value) {
    if (!pending_leaf) return;

    Node* leaf = pending_leaf;

    int term = cached_terminal_value(leaf);


    if (term != 0) {
        // Terminal node: ignore NN priors/value, back up true outcome.
        // term is from leaf->player's perspective.
        backup(leaf, term);
    } else {
        // Non-terminal: expand with NN priors and back up NN value.
        expand(leaf, priors);
        backup(leaf, value);
    }

    pending_leaf = nullptr;
}


// Best move from root after all sims
int best_action() {
    if (!root) return 0;

    if (!root->expanded || root->children.empty()) {
        // no children (e.g. zero sims); pick first legal move
        for (int i=0; i<root->Nsize*root->Nsize; i++)
            if (root->board[i] == 0)
                return i;
        return 0;
    }

    int bestA = root->children[0]->action_from_parent;
    int bestN = root->children[0]->visits;

    for (int i=1; i<(int)root->children.size(); i++) {
        if (root->children[i]->visits > bestN) {
            bestN = root->children[i]->visits;
            bestA = root->children[i]->action_from_parent;
        }
    }
    return bestA;
}

// New API functions to control GRAVE parameters
void set_grave_enabled(int enabled) {
    USE_GRAVE = (enabled != 0);
}

void set_grave_ref(double ref) {
    REF = ref;
}

void set_c_puct(double c) {
    CP = c;
}

} // extern "C"