from heapq import heappop, heappush

def dijkstra(adj, source):
    """
    Computes the shortest path distances from a source vertex to all other vertices
    in a graph using Dijkstra's algorithm (with a min-heap priority queue).

    Parameters:
    adj (List[List[Tuple[int, int]]]): Adjacency list where adj[u] contains (v, w) tuples
                                       representing an edge from u to v with weight w.
    source (int): The index of the source vertex.

    Returns:
    List[float]: A list where the ith element is the shortest distance from the source to vertex i.
                 If a vertex is unreachable, its distance will be float('inf').

    Time Complexity:
    O((n + e) * log n), where n is the number of vertices and e is the number of edges.

    Space Complexity:
    O(n + e) for the adjacency list and O(n) for the distance array.
    """
    n = len(adj)
    dist = [float('inf')] * n
    dist[source] = 0
    heap = [(0, source)]

    while heap:
        d, u = heappop(heap)
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heappush(heap, (dist[v], v))

    return dist



def bellman_ford(adj, source):
    """
    Computes the shortest path distances from a source vertex to all other vertices
    using the Bellman-Ford algorithm. Can detect negative weight cycles.

    Parameters:
    adj (List[List[Tuple[int, int]]]): Adjacency list where adj[u] contains (v, w) tuples
                                       representing an edge from u to v with weight w.
    source (int): The index of the source vertex.

    Returns:
    List[float] | None: A list of shortest distances from the source to each vertex.
                        If a negative weight cycle is detected, returns None.

    Time Complexity:
    O(n * e), where n is the number of vertices and e is the number of edges.

    Space Complexity:
    O(n + e) for the adjacency list and O(n) for the distance array.
    """
    n = len(adj)
    dist = [float('inf')] * n
    dist[source] = 0

    for _ in range(n - 1):
        for u in range(n):
            for v, w in adj[u]:
                if dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w

    for u in range(n):
        for v, w in adj[u]:
            if dist[u] + w < dist[v]:
                return None

    return dist



def floyd_warshall(adj):
    """
    Computes the shortest path distances between all pairs of vertices using the
    Floyd-Warshall algorithm.

    Parameters:
    adj (List[List[Tuple[int, int]]]): Adjacency list where adj[u] contains (v, w) tuples
                                       representing an edge from u to v with weight w.

    Returns:
    List[List[float]]: A 2D list where dist[i][j] is the shortest distance from vertex i to j.
                       If j is unreachable from i, the value is float('inf').

    Time Complexity:
    O(n^3), where n is the number of vertices.

    Space Complexity:
    O(n^2), for the distance matrix.
    """
    n = len(adj)
    dist = [[float('inf')] * n for _ in range(n)]

    for u in range(n):
        dist[u][u] = 0
        for v, w in adj[u]:
            dist[u][v] = min(dist[u][v], w)

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] < float('inf') and dist[k][j] < float('inf'):
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

    return dist


from heapq import heappush, heappop
from collections import defaultdict

def prims_algorithm(n, edges):
    """
    Computes the Minimum Spanning Tree (MST) of an undirected weighted graph using Prim's algorithm.

    Parameters:
        n (int): Number of vertices (assumed to be labeled from 0 to n-1).
        edges (List[Tuple[int, int, int]]): List of edges, where each edge is a tuple (u, v, weight)
                                            representing an undirected edge between nodes u and v.

    Returns:
        Tuple[int, List[Tuple[int, int, int]]]: 
            - mst_weight: Total weight of the Minimum Spanning Tree.
            - mst_edges: List of edges (u, v, weight) that are part of the MST.

            If the graph is not connected, returns (None, []).

    Time Complexity:
        O((n + e) * log n), where n is the number of vertices and e is the number of edges.

    Space Complexity:
        O(n + e) for the adjacency list and heap storage.
    """
    graph = defaultdict(list)
    for u, v, weight in edges:
        graph[u].append((weight, v))
        graph[v].append((weight, u))

    visited = [False] * n
    min_heap = [(0, 0, -1)]
    mst_weight = 0
    mst_edges = []

    while min_heap:
        weight, u, parent = heappop(min_heap)
        if visited[u]:
            continue
        visited[u] = True
        mst_weight += weight
        if parent != -1:
            mst_edges.append((parent, u, weight))

        for edge_weight, v in graph[u]:
            if not visited[v]:
                heappush(min_heap, (edge_weight, v, u))

    if not all(visited):
        return None, []
    
    return mst_weight, mst_edges

from collections import deque
def topo_sort_bfs(n, adj):
    """
    Return topological order of DAG using BFS (Kahn's algorithm)
    Time Complexity: O(n+m)
    Space Complexity: O(n) for indegree + queue
    """
    indeg = [0] * n
    for u in range(n):
        for v in adj[u]:
            indeg[v] += 1

    q = deque([u for u in range(n) if indeg[u] == 0])
    topo = []

    while q:
        u = q.popleft()
        topo.append(u)
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)

    return topo 


class DSU:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [1] * n
        self.size = [1] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def unite(self, x, y):
        x = self.find(x)
        y = self.find(y)
        if x != y:
            if self.rank[x] < self.rank[y]:
                x, y = y, x 
            self.parent[y] = x
            self.size[x] += self.size[y]
            if self.rank[x] == self.rank[y]:
                self.rank[x] += 1

    def cs(self, x):
        return self.size[self.find(x)]

def kruskal(n, edges):
    """
    Kruskal's algorithm to find the Minimum Spanning Tree (MST) of a graph.
    
    Args:
        n (int): The number of vertices in the graph.
        edges (list of tuples): A list of edges where each edge is represented as 
                                 a tuple (weight, vertex1, vertex2).
                                 
    Returns:
        tuple: A tuple containing the total weight of the MST and a list of the edges in the MST.
        
    Time Complexity:
        - Sorting edges: O(E log E), where E is the number of edges.
        - Union/Find operations: O(E * α(n)), where α is the inverse Ackermann function (almost constant).
        
    Space Complexity:
        - O(V + E), where V is the number of vertices and E is the number of edges (for DSU and edges list).
    """
    # Initialize DSU to manage the connected components
    dsu = DSU(n)
    mst_weight = 0  # Total weight of the MST
    mst_edges = []  # Edges included in the MST

    # Step 1: Sort the edges by weight (ascending)
    edges.sort()  # Sorting edges based on weight (first element of each tuple)

    # Step 2: Process each edge in sorted order
    for weight, u, v in edges:
        if dsu.find(u) != dsu.find(v):  # If u and v are in different components
            dsu.unite(u, v)  # Unite the components
            mst_weight += weight  # Add the edge's weight to the MST weight
            mst_edges.append((u, v, weight))  # Add the edge to the MST

    return mst_weight, mst_edges
