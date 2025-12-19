#include <vector>
#include <queue>
#include <cmath>
#include <cstring>
#include <algorithm>

struct Node {
    std::vector<int> board;      
    int player;                  
    int Nsize;                   
    Node* parent;
    int action_from_parent;     


    int visits;
    double value_sum;
    double Q;                    

    int virtual_visits;


    std::vector<int> amaf_visits;      
    std::vector<double> amaf_value;    
    std::vector<double> Q_amaf;        


    bool expanded;
    std::vector<Node*> children;
    std::vector<int> children_actions;   
    std::vector<double> priors;          

    Node(const std::vector<int>& b, int p, int n, Node* par, int act)
        : board(b),
          player(p),
          Nsize(n),
          parent(par),
          action_from_parent(act),
          visits(0),
          value_sum(0.0),
          Q(0.0),
          virtual_visits(0),
          expanded(false)
    {
        amaf_visits.resize(n * n, 0);
        amaf_value.resize(n * n, 0.0);
        Q_amaf.resize(n * n, 0.0);
    }
};



static Node* root = nullptr;
static Node* pending_leaf = nullptr;
static std::vector<Node*> pending_leaves;

static int BOARD_N = 11;
static double CP = 1.2;
static double REF = 0.5;   
static bool USE_GRAVE = true;


static inline bool inside_xy(int x, int y, int N) {
    return x >= 0 && x < N && y >= 0 && y < N;
}

bool red_wins(const std::vector<int>& b, int N) {
    std::queue<int> q;
    std::vector<char> visited(N * N, 0);

    for (int y = 0; y < N; y++) {
        if (b[y] == 1) {
            q.push(y);
            visited[y] = 1;
        }
    }

    int dx[6] = {-1, -1, 0, 1, 1, 0};
    int dy[6] = {0, 1, 1, 0, -1, -1};

    while (!q.empty()) {
        int idx = q.front();
        q.pop();
        int x = idx / N;
        int y = idx % N;

        if (x == N - 1) return true;

        for (int k = 0; k < 6; k++) {
            int nx = x + dx[k], ny = y + dy[k];
            if (!inside_xy(nx, ny, N)) continue;
            int id2 = nx * N + ny;
            if (!visited[id2] && b[id2] == 1) {
                visited[id2] = 1;
                q.push(id2);
            }
        }
    }
    return false;
}

bool blue_wins(const std::vector<int>& b, int N) {
    std::queue<int> q;
    std::vector<char> visited(N * N, 0);

    for (int x = 0; x < N; x++) {
        int idx = x * N + 0;
        if (b[idx] == 2) {
            q.push(idx);
            visited[idx] = 1;
        }
    }

    int dx[6] = {-1, -1, 0, 1, 1, 0};
    int dy[6] = {0, 1, 1, 0, -1, -1};

    while (!q.empty()) {
        int idx = q.front();
        q.pop();
        int x = idx / N;
        int y = idx % N;

        if (y == N - 1) return true;

        for (int k = 0; k < 6; k++) {
            int nx = x + dx[k], ny = y + dy[k];
            if (!inside_xy(nx, ny, N)) continue;
            int id2 = nx * N + ny;
            if (!visited[id2] && b[id2] == 2) {
                visited[id2] = 1;
                q.push(id2);
            }
        }
    }
    return false;
}

int check_terminal_value(const std::vector<int>& b, int player, int N) {
    bool r = red_wins(b, N);
    bool bl = blue_wins(b, N);

    if (!r && !bl) return 0;

    int winner = 0;
    if (r && !bl) winner = 1;
    else if (bl && !r) winner = 2;
    else return 0; 

    return (player == winner) ? +1 : -1;
}




double compute_beta(int n_visits, double k = 2000.0) {
    return std::sqrt(k / (3.0 * n_visits + k));
}



static inline void add_virtual(Node* leaf, int vl = 1) {
    for (Node* n = leaf; n != nullptr; n = n->parent) {
        n->virtual_visits += vl;
    }
}

static inline void remove_virtual(Node* leaf, int vl = 1) {
    for (Node* n = leaf; n != nullptr; n = n->parent) {
        n->virtual_visits -= vl;
        if (n->virtual_visits < 0) n->virtual_visits = 0;
    }
}



Node* select_leaf(Node* node) {
    while (node->expanded && !node->children.empty()) {
        double totalN = 0.0;
        for (Node* c : node->children) {
            totalN += (double)(c->visits + c->virtual_visits);
        }
        if (totalN < 1e-8) totalN = 1e-8;

        double best = -1e18;
        int best_i = 0;

        for (int i = 0; i < (int)node->children.size(); i++) {
            Node* c = node->children[i];
            int action = node->children_actions[i];

            int cN = c->visits + c->virtual_visits;

            double Q_mc = -c->Q;

            double Q_combined = Q_mc;

            if (USE_GRAVE && cN > 0) {
                double Q_amaf_val = 0.0;
                if (node->amaf_visits[action] > 0) {
                    Q_amaf_val = node->Q_amaf[action];
                }

                double beta = compute_beta(cN);
                Q_combined = (1.0 - beta) * Q_mc + beta * (Q_amaf_val + REF);
            }

            double U = CP * node->priors[i] * std::sqrt(totalN) / (1.0 + (double)cN);
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


void expand(Node* leaf, const double* priors) {
    if (leaf->expanded) return;

    int N = leaf->Nsize;
    std::vector<int> legal;
    legal.reserve(N * N);

    for (int i = 0; i < N * N; i++) {
        if (leaf->board[i] == 0) legal.push_back(i);
    }

    leaf->expanded = true;
    leaf->children_actions = legal;

    int L = (int)legal.size();
    leaf->priors.resize(L);
    leaf->children.resize(L, nullptr);

    for (int i = 0; i < L; i++) {
        leaf->priors[i] = priors[legal[i]];
    }

    for (int i = 0; i < L; i++) {
        int a = legal[i];
        std::vector<int> nb = leaf->board;
        nb[a] = leaf->player;

        int nextp = (leaf->player == 1 ? 2 : 1);
        leaf->children[i] = new Node(nb, nextp, N, leaf, a);
    }
}


void backup(Node* leaf, double value) {
    std::vector<int> action_sequence;
    std::vector<int> player_sequence;

    Node* cur = leaf;
    while (cur != nullptr && cur->parent != nullptr) {
        action_sequence.push_back(cur->action_from_parent);
        player_sequence.push_back(cur->parent->player);
        cur = cur->parent;
    }

    std::reverse(action_sequence.begin(), action_sequence.end());
    std::reverse(player_sequence.begin(), player_sequence.end());

    cur = leaf;
    double v = value;

    while (cur != nullptr) {
        cur->visits++;
        cur->value_sum += v;
        cur->Q = cur->value_sum / cur->visits;

        if (USE_GRAVE && cur->parent != nullptr) {
            int cur_depth = 0;
            Node* tmp = cur;
            while (tmp->parent != nullptr) {
                cur_depth++;
                tmp = tmp->parent;
            }

            for (size_t j = (size_t)cur_depth; j < action_sequence.size(); j++) {
                int action = action_sequence[j];
                int action_player = player_sequence[j];

                if (action_player == cur->player) {
                    cur->amaf_visits[action]++;
                    cur->amaf_value[action] += v;
                    cur->Q_amaf[action] = cur->amaf_value[action] / cur->amaf_visits[action];
                }
            }
        }

        v = -v;
        cur = cur->parent;
    }
}



void free_tree(Node* n) {
    if (!n) return;
    for (auto c : n->children) free_tree(c);
    delete n;
}

void init_root(const int* board, int N, int player) {
    BOARD_N = N;

    std::vector<int> b(board, board + N * N);

    if (root) free_tree(root);
    root = new Node(b, player, N, nullptr, -1);

    pending_leaf = nullptr;
    pending_leaves.clear();
}



extern "C" {

void request_leaves(int batch_size, int* out_boards, int* out_players, int* out_is_terminal) {
    if (!root) {
        for (int i = 0; i < batch_size; i++) out_is_terminal[i] = 1;
        return;
    }

    int NN = root->Nsize * root->Nsize;
    pending_leaves.assign(batch_size, nullptr);

    for (int i = 0; i < batch_size; i++) {
        Node* leaf = select_leaf(root);
        pending_leaves[i] = leaf;

        add_virtual(leaf, 1);

        int term = check_terminal_value(leaf->board, leaf->player, leaf->Nsize);
        out_is_terminal[i] = (term != 0);
        out_players[i] = leaf->player;

        int base = i * NN;
        for (int k = 0; k < NN; k++) out_boards[base + k] = leaf->board[k];
    }
}

void apply_evals_batch(int batch_size, const double* priors_batch, const double* values_batch) {
    if ((int)pending_leaves.size() != batch_size) return;
    if (!root) return;

    int NN = root->Nsize * root->Nsize;

    for (int i = 0; i < batch_size; i++) {
        Node* leaf = pending_leaves[i];
        if (!leaf) continue;

        remove_virtual(leaf, 1);

        int term = check_terminal_value(leaf->board, leaf->player, leaf->Nsize);

        if (term != 0) {
            backup(leaf, term);
        } else {
            const double* pri = priors_batch + i * NN;
            expand(leaf, pri);
            backup(leaf, values_batch[i]);
        }
    }

    pending_leaves.clear();
}

void reset_tree(int* board, int N, int player) {
    init_root(board, N, player);
}

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
    for (int i = 0; i < NN; i++) out_board[i] = leaf->board[i];
    *out_player = leaf->player;
}

void apply_eval(const double* priors, double value) {
    if (!pending_leaf) return;

    Node* leaf = pending_leaf;

    int term = check_terminal_value(leaf->board, leaf->player, leaf->Nsize);

    if (term != 0) {
        backup(leaf, term);
    } else {
        expand(leaf, priors);
        backup(leaf, value);
    }

    pending_leaf = nullptr;
}

int best_action() {
    if (!root) return 0;

    if (!root->expanded || root->children.empty()) {
        for (int i = 0; i < root->Nsize * root->Nsize; i++) {
            if (root->board[i] == 0) return i;
        }
        return 0;
    }

    int bestA = root->children[0]->action_from_parent;
    int bestN = root->children[0]->visits;

    for (int i = 1; i < (int)root->children.size(); i++) {
        if (root->children[i]->visits > bestN) {
            bestN = root->children[i]->visits;
            bestA = root->children[i]->action_from_parent;
        }
    }
    return bestA;
}

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
