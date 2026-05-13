#include <bits/stdc++.h>
using namespace std;

using ll = long long;
const ll INFLL = (1LL << 62);

struct DSU {
    vector<int> parent, rank_, size_;

    DSU(int n = 0) { init(n); }

    void init(int n) {
        parent.resize(n);
        iota(parent.begin(), parent.end(), 0);
        rank_.assign(n, 1);
        size_.assign(n, 1);
    }

    int find(int x) {
        if (parent[x] != x) parent[x] = find(parent[x]);
        return parent[x];
    }

    bool unite(int x, int y) {
        x = find(x);
        y = find(y);
        if (x == y) return false;
        if (rank_[x] < rank_[y]) swap(x, y);
        parent[y] = x;
        size_[x] += size_[y];
        if (rank_[x] == rank_[y]) rank_[x]++;
        return true;
    }

    int cs(int x) {
        return size_[find(x)];
    }
};

struct WeightedDSU {
    vector<int> parent;
    vector<ll> dist; // dist[i] = value(i) - value(parent[i])

    WeightedDSU(int n = 0) { init(n); }

    void init(int n) {
        parent.resize(n + 1);
        iota(parent.begin(), parent.end(), 0);
        dist.assign(n + 1, 0);
    }

    int find(int x) {
        if (parent[x] == x) return x;
        int p = parent[x];
        int root = find(p);
        dist[x] += dist[p];
        return parent[x] = root;
    }

    bool relate(int a, int b, ll d) {
        int ra = find(a), rb = find(b);
        if (ra != rb) {
            parent[ra] = rb;
            dist[ra] = d + dist[b] - dist[a];
            return true;
        }
        return dist[a] - dist[b] == d;
    }

    optional<ll> get_dist(int a, int b) {
        if (find(a) != find(b)) return nullopt;
        return dist[a] - dist[b];
    }
};

vector<ll> dijkstra(const vector<vector<pair<int, ll>>>& adj, int source) {
    int n = (int)adj.size();
    vector<ll> dist(n, INFLL);
    priority_queue<pair<ll, int>, vector<pair<ll, int>>, greater<pair<ll, int>>> pq;
    dist[source] = 0;
    pq.push({0, source});

    while (!pq.empty()) {
        auto [d, u] = pq.top();
        pq.pop();
        if (d != dist[u]) continue;
        for (auto [v, w] : adj[u]) {
            if (dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                pq.push({dist[v], v});
            }
        }
    }
    return dist;
}

optional<vector<ll>> bellman_ford(const vector<vector<pair<int, ll>>>& adj, int source) {
    int n = (int)adj.size();
    vector<ll> dist(n, INFLL);
    dist[source] = 0;

    for (int it = 0; it < n - 1; it++) {
        bool changed = false;
        for (int u = 0; u < n; u++) {
            if (dist[u] == INFLL) continue;
            for (auto [v, w] : adj[u]) {
                if (dist[u] + w < dist[v]) {
                    dist[v] = dist[u] + w;
                    changed = true;
                }
            }
        }
        if (!changed) break;
    }

    for (int u = 0; u < n; u++) {
        if (dist[u] == INFLL) continue;
        for (auto [v, w] : adj[u]) {
            if (dist[u] + w < dist[v]) return nullopt;
        }
    }
    return dist;
}

vector<vector<ll>> floyd_warshall(const vector<vector<pair<int, ll>>>& adj) {
    int n = (int)adj.size();
    vector<vector<ll>> dist(n, vector<ll>(n, INFLL));
    for (int u = 0; u < n; u++) {
        dist[u][u] = 0;
        for (auto [v, w] : adj[u]) dist[u][v] = min(dist[u][v], w);
    }

    for (int k = 0; k < n; k++) {
        for (int i = 0; i < n; i++) {
            if (dist[i][k] == INFLL) continue;
            for (int j = 0; j < n; j++) {
                if (dist[k][j] == INFLL) continue;
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j]);
            }
        }
    }
    return dist;
}

pair<optional<ll>, vector<tuple<int, int, ll>>> prims_algorithm(int n, const vector<tuple<int, int, ll>>& edges) {
    vector<vector<pair<int, ll>>> graph(n);
    for (auto [u, v, w] : edges) {
        graph[u].push_back({v, w});
        graph[v].push_back({u, w});
    }

    vector<bool> visited(n, false);
    priority_queue<tuple<ll, int, int>, vector<tuple<ll, int, int>>, greater<tuple<ll, int, int>>> pq;
    pq.push({0, 0, -1});
    ll mst_weight = 0;
    vector<tuple<int, int, ll>> mst_edges;

    while (!pq.empty()) {
        auto [w, u, parent] = pq.top();
        pq.pop();
        if (visited[u]) continue;
        visited[u] = true;
        mst_weight += w;
        if (parent != -1) mst_edges.push_back({parent, u, w});
        for (auto [v, edge_weight] : graph[u]) {
            if (!visited[v]) pq.push({edge_weight, v, u});
        }
    }

    if (any_of(visited.begin(), visited.end(), [](bool x) { return !x; })) return {nullopt, {}};
    return {mst_weight, mst_edges};
}

vector<int> topo_sort_bfs(int n, const vector<vector<int>>& adj) {
    vector<int> indeg(n, 0), topo;
    for (int u = 0; u < n; u++) {
        for (int v : adj[u]) indeg[v]++;
    }

    queue<int> q;
    for (int u = 0; u < n; u++) {
        if (indeg[u] == 0) q.push(u);
    }

    while (!q.empty()) {
        int u = q.front();
        q.pop();
        topo.push_back(u);
        for (int v : adj[u]) {
            if (--indeg[v] == 0) q.push(v);
        }
    }
    return topo;
}
