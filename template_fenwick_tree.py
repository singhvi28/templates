class FenwickTree:
    """
    Fenwick Tree (Binary Indexed Tree) for prefix sums.
    Supports:
        - add(index, delta)        : Point update
        - sum(index)               : Prefix sum query [1..index]
        - range_sum(l, r)          : Sum in [l..r]
        - find_kth(k)              : Smallest idx with prefix_sum(idx) >= k  (binary lifting)
        - lower_bound(target)      : Smallest idx with prefix_sum(idx) >= target
        - upper_bound(target)      : Smallest idx with prefix_sum(idx) > target
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

    def range_sum(self, l, r):
        """Return sum in range [l, r]."""
        return self.sum(r) - self.sum(l - 1)

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

    def lower_bound(self, target):
        """
        Return smallest idx such that prefix_sum(idx) >= target.
        Same as find_kth(target).
        """
        return self.find_kth(target)

    def upper_bound(self, target):
        """
        Return smallest idx such that prefix_sum(idx) > target.
        Equivalent to find_kth(target + 1).
        """
        return self.find_kth(target + 1)