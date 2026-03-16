class SegTree:
    """Iterative segment tree supporting point updates and range queries [l, r)."""

    def __init__(self, n, op, e):
        """
        Initialise tree for n elements.

        Parameters
        ----------
        n : int
            Number of elements.
        op : function
            Binary associative operation (e.g. sum, min, xor).
        e : any
            Identity element of the operation.
        """
        self.n = n
        self.op = op
        self.e = e

        self.size = 1 << (n - 1).bit_length()
        self.d = [e] * (2 * self.size)

    def build(self, arr):
        """Build tree from initial array."""
        for i in range(len(arr)):
            self.d[self.size + i] = arr[i]

        for i in range(self.size - 1, 0, -1):
            self.d[i] = self.op(self.d[2*i], self.d[2*i+1])

    def set(self, p, x):
        """Set value at index p to x."""
        p += self.size
        self.d[p] = x

        while p > 1:
            p >>= 1
            self.d[p] = self.op(self.d[2*p], self.d[2*p+1])

    def get(self, p):
        """Return value at index p."""
        return self.d[p + self.size]

    def prod(self, l, r):
        """Return operation result over range [l, r)."""
        sml = self.e
        smr = self.e

        l += self.size
        r += self.size

        while l < r:
            if l & 1:
                sml = self.op(sml, self.d[l])
                l += 1
            if r & 1:
                r -= 1
                smr = self.op(self.d[r], smr)

            l >>= 1
            r >>= 1

        return self.op(sml, smr)

# EXAMPLE USAGE
# def op(a, b):
#     return a ^ b
# arr = [1, 2, 3, 4]
# seg = SegTree(len(arr), op, 0)
# seg.build(arr)
# print(seg.prod(1, 4))  # XOR on [1,4)
# seg.set(2, 10)
# print(seg.prod(1, 4))

from bisect import bisect_right

class MST:
    def __init__(self, a):
        self.n = len(a)
        self.v = [[] for _ in range(4 * self.n)]
        self.p = [[] for _ in range(4 * self.n)]
        if self.n > 0:
            self._build(1, 0, self.n - 1, a)

    def _build(self, idx, l, r, a):
        if l == r:
            val = a[l]
            self.v[idx] = [val]
            self.p[idx] = [val]
            return
        m = (l + r) // 2
        self._build(idx * 2, l, m, a)
        self._build(idx * 2 + 1, m + 1, r, a)
        L = self.v[idx * 2]
        R = self.v[idx * 2 + 1]
        merged = []
        i = j = 0
        while i < len(L) and j < len(R):
            if L[i] <= R[j]:
                merged.append(L[i])
                i += 1
            else:
                merged.append(R[j])
                j += 1
        if i < len(L):
            merged.extend(L[i:])
        if j < len(R):
            merged.extend(R[j:])
        self.v[idx] = merged

        prefix = []
        s = 0
        for val in merged:
            s += val
            prefix.append(s)
        self.p[idx] = prefix

    # internal version
    def _le(self, idx, l, r, L, R, x):
        if R < l or r < L:
            return 0, 0
        if L <= l and r <= R:
            A = self.v[idx]
            P = self.p[idx]
            pos = bisect_right(A, x)  # first index > x
            s = P[pos - 1] if pos > 0 else 0
            return pos, s
        m = (l + r) // 2
        a_cnt, a_sum = self._le(idx * 2, l, m, L, R, x)
        b_cnt, b_sum = self._le(idx * 2 + 1, m + 1, r, L, R, x)
        return a_cnt + b_cnt, a_sum + b_sum

    # public wrapper: query [L, R], values <= x
    def le(self, L, R, x):
        if self.n == 0:
            return 0, 0
        return self._le(1, 0, self.n - 1, L, R, x)

    # internal version
    def _sum(self, idx, l, r, L, R):
        if R < l or r < L:
            return 0
        if L <= l and r <= R:
            return self.p[idx][-1] if self.p[idx] else 0
        m = (l + r) // 2
        return self._sum(idx * 2, l, m, L, R) + self._sum(idx * 2 + 1, m + 1, r, L, R)

    # public wrapper: range sum on [L, R]
    def range_sum(self, L, R):
        if self.n == 0:
            return 0
        return self._sum(1, 0, self.n - 1, L, R)
