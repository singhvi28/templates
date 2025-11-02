class FenwickTree:
    """
    Fenwick Tree (Binary Indexed Tree) for prefix sums.
    Supports:
        - add(index, delta)        : Point update
        - sum(index)               : Prefix sum query [1..index]
        - range_sum(l, r)          : Sum in [l..r]
        - find_kth(k)              : Smallest idx with prefix_sum(idx) >= k  (binary lifting)
    Indexing: 1-based for internal operations.
        
    Authored By: akkisinghvi28
    
    """

    def __init__(self, n):
        """Initialise Fenwick Tree for n elements (1-based indexing)."""
        self.n = n
        self.bit = [0] * (n + 1)

    def add(self, idx, delta):
        """Add `delta` to element at position `idx`."""
        while idx <= self.n:
            self.bit[idx] += delta
            idx += idx & -idx

    def sum(self, idx):
        """Return prefix sum from 1 to idx."""
        res = 0
        while idx > 0:
            res += self.bit[idx]
            idx -= idx & -idx
        return res

    def find_kth(self, k):
        """
        Binary lifting: Find smallest index such that prefix_sum(index) >= k.
        If total sum < k, returns n+1 (invalid).
        """
        idx = 0
        bit_mask = 1 << (self.n.bit_length() - 1)
        while bit_mask:
            nxt = idx + bit_mask
            if nxt <= self.n and self.bit[nxt] < k:
                k -= self.bit[nxt]
                idx = nxt
            bit_mask >>= 1
        return idx + 1



class FenwickTree2D:
    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.bit = [[0] * (m + 1) for _ in range(n + 1)]

    def add(self, x, y, delta):
        i = x
        while i <= self.n:
            j = y
            while j <= self.m:
                self.bit[i][j] += delta
                j += j & -j
            i += i & -i

    def sum(self, x, y):
        res = 0
        i = x
        while i > 0:
            j = y
            while j > 0:
                res += self.bit[i][j]
                j -= j & -j
            i -= i & -i
        return res

    def range_sum(self, x1, y1, x2, y2):
        return (self.sum(x2, y2)
                - self.sum(x1 - 1, y2)
                - self.sum(x2, y1 - 1)
                + self.sum(x1 - 1, y1 - 1))
