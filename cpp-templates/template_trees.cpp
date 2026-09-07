#include <bits/stdc++.h>
using namespace std;

struct LCA {
    int n, LOG;
    vector<int> a, lvl;
    vector<vector<int>> par, mx, adj;

    LCA() : n(0), LOG(0) {}

    LCA(int n_, const vector<int>& values, const vector<pair<int, int>>& edges, int root = 1) {
        init(n_, values, edges, root);
    }

    void init(int n_, const vector<int>& values, const vector<pair<int, int>>& edges, int root = 1) {
        n = n_;
        a.assign(n + 1, 0);
        for (int i = 1; i <= n; i++) a[i] = values[i - 1];
        LOG = 1;
        while ((1 << LOG) <= n + 1) LOG++;
        lvl.assign(n + 1, 0);
        par.assign(n + 1, vector<int>(LOG, 0));
        mx.assign(n + 1, vector<int>(LOG, 0));
        adj.assign(n + 1, {});
        for (auto [u, v] : edges) {
            adj[u].push_back(v);
            adj[v].push_back(u);
        }
        dfs(root, 0);
        build();
    }

    void dfs(int root, int parent) {
        stack<pair<int, int>> st;
        st.push({root, parent});
        while (!st.empty()) {
            auto [u, p] = st.top();
            st.pop();
            lvl[u] = lvl[p] + 1;
            par[u][0] = p;
            mx[u][0] = a[u];
            for (int v : adj[u]) {
                if (v != p) st.push({v, u});
            }
        }
    }

    void build() {
        for (int j = 1; j < LOG; j++) {
            for (int i = 1; i <= n; i++) {
                int mid = par[i][j - 1];
                par[i][j] = par[mid][j - 1];
                mx[i][j] = max(mx[i][j - 1], mx[mid][j - 1]);
            }
        }
    }

    int lift(int u, int k) const {
        for (int j = 0; j < LOG; j++) {
            if ((k >> j) & 1) u = par[u][j];
        }
        return u;
    }

    int get_max(int u, int k) const {
        int res = 0;
        for (int j = 0; j < LOG; j++) {
            if ((k >> j) & 1) {
                res = max(res, mx[u][j]);
                u = par[u][j];
            }
        }
        return res;
    }

    int lca(int u, int v) const {
        if (lvl[u] < lvl[v]) swap(u, v);
        u = lift(u, lvl[u] - lvl[v]);
        if (u == v) return u;
        for (int j = LOG - 1; j >= 0; j--) {
            if (par[u][j] != par[v][j]) {
                u = par[u][j];
                v = par[v][j];
            }
        }
        return par[u][0];
    }

    int dist(int u, int v) const {
        int l = lca(u, v);
        return lvl[u] + lvl[v] - 2 * lvl[l];
    }

    int get_max_on_path(int u, int v) const {
        int l = lca(u, v);
        return max({get_max(u, lvl[u] - lvl[l]), get_max(v, lvl[v] - lvl[l]), a[l]});
    }
};

// Small-to-large merging pattern for subtree maps.
map<int, int>* dfs_small_to_large(int u, int p, const vector<vector<int>>& adj, const vector<int>& values,
                                  vector<int>& results) {
    auto* main_container = new map<int, int>();
    (*main_container)[values[u]] = 1;

    for (int v : adj[u]) {
        if (v == p) continue;
        map<int, int>* child = dfs_small_to_large(v, u, adj, values, results);
        if (main_container->size() < child->size()) swap(main_container, child);
        for (auto [key, val] : *child) (*main_container)[key] += val;
        delete child;
    }

    results[u] = (int)main_container->size();
    return main_container;
}

// Euler Tour technique mapping subtree queries to contiguous range queries
tuple<vector<int>, vector<int>, vector<int>> eulerTour(const vector<vector<int>>& adj, int root = 0) {
    int n = (int)adj.size();
    vector<int> tin(n, 0), tout(n, 0), tour;
    int timer = 0;

    // stack stores {node, parent, next_child_idx}
    vector<tuple<int, int, int>> stack;
    stack.push_back({root, -1, 0});

    tin[root] = timer++;
    tour.push_back(root);

    while (!stack.empty()) {
        auto& [u, p, i] = stack.back();

        if (i == (int)adj[u].size()) {
            tout[u] = timer - 1;
            stack.pop_back();
            if (!stack.empty()) {
                tour.push_back(get<0>(stack.back()));
            }
            continue;
        }

        int v = adj[u][i++];
        if (v == p) continue;

        tin[v] = timer++;
        tour.push_back(v);
        stack.push_back({v, u, 0});
    }

    return {tour, tin, tout};
}
