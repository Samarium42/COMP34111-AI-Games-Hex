#include <vector>
#include <cmath>
#include <cstring>
#include <unordered_map>
#include <algorithm>
#include <cstdlib>

// =========================
// Node structure
// =========================

struct Node {
    std::vector<int> board;       // flat N*N board (0 empty, 1 red, 2 blue)
    int player;                   // player to move (1 or 2)
    int Nsize;                    // board size
    Node* parent;                 // parent pointer
    int action_from_parent;       // action index

    // MCTS stats
    int N;                        // visit count
    double W;                     // total value
    double Q;                     // mean value

    // children
    std::vector<Node*> children;
    std::vector<int> children_actions;
    std::vector<double> priors;

    Node(const std::vector<int>& b, int p, int nsize, Node* par, int act)
        : board(b),
          player(p),
          Nsize(nsize),
          parent(par),
          action_from_parent(act),
          N(0),
          W(0.0),
          Q(0.0)
    {}
};


// =========================
// Global MCTS state
// =========================

static Node* root = nullptr;
static Node* pending_leaf = nullptr;

static int BOARD_N = 11;
static double CP = 1.2;
static int SIMS = 300;


// =========================
// Selection
// =========================

Node* select_leaf(Node* node) {
    while (!node->children.empty()) {
        int m = node->children.size();

        double totalN = 0.0;
        for (auto c : node->children) totalN += c->N;

        double best_score = -1e18;
        int best_i = 0;

        for (int i = 0; i < m; i++) {
            Node* c = node->children[i];

            double Q = c->Q;
            double U = CP * node->priors[i] * std::sqrt(totalN + 1e-8) / (1 + c->N);

            double score = Q + U;
            if (score > best_score) {
                best_score = score;
                best_i = i;
            }
        }

        node = node->children[best_i];
    }

    return node;
}


// =========================
// Backup
// =========================

void backup(Node* leaf, double value) {
    Node* cur = leaf;
    double v = value;

    while (cur != nullptr) {
        cur->N += 1;
        cur->W += v;
        cur->Q = cur->W / cur->N;
        v = -v;
        cur = cur->parent;
    }
}


// =========================
// Root initialization
// =========================

void init_root(const int* board, int player, int N) {
    BOARD_N = N;
    std::vector<int> b(board, board + N * N);

    if (root) delete root;   // NOTE: simple free, not recursive
    root = new Node(b, player, N, nullptr, -1);
}


// =========================
// ====== C API =========
// =========================

extern "C" {


// reset_tree() : called at the start of each move
void reset_tree(int* board, int N, int player) {
    init_root(board, player, N);
}


// request_leaf() : Python asks C++ for one leaf to evaluate
void request_leaf(int* out_board,
                  int* out_player,
                  int* out_is_terminal)
{
    if (!root) {
        *out_is_terminal = 1;
        return;
    }

    Node* leaf = select_leaf(root);
    pending_leaf = leaf;

    int NN = leaf->Nsize * leaf->Nsize;
    for (int i = 0; i < NN; i++)
        out_board[i] = leaf->board[i];

    *out_player = leaf->player;

    // Terminal detection: Python handles this anyway
    bool any_empty = false;
    for (int v : leaf->board)
        if (v == 0) { any_empty = true; break; }

    *out_is_terminal = any_empty ? 0 : 1;
}


// apply_eval() : Python gives C++ priors + value
void apply_eval(const double* priors, double value) {
    Node* leaf = pending_leaf;
    if (!leaf) return;

    int N = leaf->Nsize;

    // Discover legal moves
    std::vector<int> legal;
    legal.reserve(N * N);

    for (int i = 0; i < N * N; i++)
        if (leaf->board[i] == 0)
            legal.push_back(i);

    // Build children
    int L = legal.size();
    leaf->children_actions = legal;
    leaf->priors.resize(L);
    leaf->children.resize(L, nullptr);

    for (int i = 0; i < L; i++)
        leaf->priors[i] = priors[legal[i]];

    for (int i = 0; i < L; i++) {
        int a = legal[i];
        std::vector<int> b2 = leaf->board;
        b2[a] = leaf->player;
        int next_p = (leaf->player == 1 ? 2 : 1);
        leaf->children[i] = new Node(b2, next_p, N, leaf, a);
    }

    backup(leaf, value);
}


// best_action() : after all NN evals, Python asks for root move
int best_action() {
    if (!root) return 0;

    int M = root->children.size();
    if (M == 0) {
        // no children = no legal moves updated
        for (int i = 0; i < root->Nsize * root->Nsize; i++)
            if (root->board[i] == 0)
                return i;
        return 0;
    }

    int best_a = root->children[0]->action_from_parent;
    int best_N = root->children[0]->N;

    for (int i = 1; i < M; i++) {
        if (root->children[i]->N > best_N) {
            best_N = root->children[i]->N;
            best_a = root->children[i]->action_from_parent;
        }
    }

    return best_a;
}


} // extern "C"
