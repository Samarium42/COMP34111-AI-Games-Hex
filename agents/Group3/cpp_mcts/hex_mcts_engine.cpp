#include <vector>
#include <queue>
#include <cmath>
#include <cstring>
#include <algorithm>

struct Node {
    std::vector<int> board;      // flat N*N board
    int player;                  // player to move (1=R, 2=B)
    int Nsize;                   // board size
    Node* parent;
    int action_from_parent;      // index 0..N*N-1

    // stats
    int visits;
    double value_sum;
    double Q;                    // value from this node's player's perspective

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
    {}
};


// =======================================================================
// Global root + settings
// =======================================================================

static Node* root = nullptr;
static Node* pending_leaf = nullptr;

static int BOARD_N = 11;
static double CP = 1.2;


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
// Selection (PUCT)
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

            // c->Q is from c->player's perspective.
            // For selection at 'node', we want value from node->player's perspective.
            double Q = -c->Q;

            double U = CP * node->priors[i] * std::sqrt(totalN) / (1.0 + c->visits);
            double score = Q + U;
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

    for (int i=0; i<L; i++) {
        int a = legal[i];
        std::vector<int> nb = leaf->board;
        nb[a] = leaf->player;

        int nextp = (leaf->player == 1 ? 2 : 1);
        leaf->children[i] = new Node(nb, nextp, N, leaf, a);
    }
}


// =======================================================================
// Backup
// value is from the leaf node's player's perspective.
// We flip sign as we go up so each node's Q is from its own player's POV.
// =======================================================================

void backup(Node* leaf, double value) {
    Node* cur = leaf;
    double v = value;

    while (cur != nullptr) {
        cur->visits++;
        cur->value_sum += v;
        cur->Q = cur->value_sum / cur->visits;

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

    std::vector<int> b(board, board + N*N);

    if (root) free_tree(root);
    root = new Node(b, player, N, nullptr, -1);
    pending_leaf = nullptr;  // ensure no stale pointer between moves
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

    int term = check_terminal_value(leaf->board, leaf->player, leaf->Nsize);
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

    int term = check_terminal_value(leaf->board, leaf->player, leaf->Nsize);

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

} // extern "C"
