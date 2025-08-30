class LCA:
    """
    Binary Lifting with LCA and maximum-on-path queries.
    """

    def __init__(self, n, values, edges, root=1):
        """
        Initialize the structure.

        Args:
            n (int): Number of nodes.
            values (list[int]): Node values (0-indexed, will be shifted to 1-indexed).
            edges (list[tuple[int, int]]): Undirected edges of the tree.
            root (int): Root of the tree (default=1).
        
        Authored By:
            akkisinghvi28
        """
        self.n = n
        self.a = [0] + values  # 1-indexed
        self.LOG = (n+1).bit_length()
        self.lvl = [0] * (n + 1)
        self.par = [[0] * self.LOG for _ in range(n + 1)]
        self.mx = [[0] * self.LOG for _ in range(n + 1)]

        self.adj = [[] for _ in range(n + 1)]
        for u, v in edges:
            self.adj[u].append(v)
            self.adj[v].append(u)

        self._dfs(root, 0)
        self._build()

    def _dfs(self, u, p):
        """DFS to set up parent and level arrays."""
        self.lvl[u] = self.lvl[p] + 1
        self.par[u][0] = p
        self.mx[u][0] = self.a[u]
        for v in self.adj[u]:
            if v == p:
                continue
            self._dfs(v, u)

    def _build(self):
        """Precompute binary lifting and maximum values."""
        for j in range(1, self.LOG):
            for i in range(1, self.n + 1):
                mid = self.par[i][j - 1]
                self.par[i][j] = self.par[mid][j - 1]
                self.mx[i][j] = max(self.mx[i][j - 1], self.mx[mid][j - 1])

    def lift(self, u, k):
        """Lift node u by k steps."""
        for j in range(self.LOG):
            if (k >> j) & 1:
                u = self.par[u][j]
        return u

    def get_max(self, u, k):
        """Get maximum value along path when lifting u by k steps."""
        res = 0
        for j in range(self.LOG):
            if (k >> j) & 1:
                res = max(res, self.mx[u][j])
                u = self.par[u][j]
        return res

    def lca(self, u, v):
        """Compute lowest common ancestor (LCA) of u and v."""
        if self.lvl[u] < self.lvl[v]:
            u, v = v, u
        u = self.lift(u, self.lvl[u] - self.lvl[v])
        if u == v:
            return u
        for j in reversed(range(self.LOG)):
            if self.par[u][j] != self.par[v][j]:
                u = self.par[u][j]
                v = self.par[v][j]
        return self.par[u][0]

    def dist(self, u, v):
        """Compute distance (#edges) between u and v."""
        l = self.lca(u, v)
        return self.lvl[u] + self.lvl[v] - 2 * self.lvl[l]

    def get_max_on_path(self, u, v):
        """Get maximum value on path between u and v."""
        l = self.lca(u, v)
        return max(self.get_max(u, self.lvl[u] - self.lvl[l]),
                   self.get_max(v, self.lvl[v] - self.lvl[l]),
                   self.a[l])

lca = LCA()