# Credits to HealthyUG for the inspiration.
# Segment Tree with Point Updates and Range Queries
# Supports multiple Segment Trees with just a change in the Node and Update
# Very few changes required every time

class SegTree:
    def __init__(self, a_len, a, Node, Update):  # change if type updated
        self.arr = a
        self.n = a_len
        self.Node = Node
        self.Update = Update
        self.s = 1
        while self.s < 2 * self.n:
            self.s <<= 1
        self.tree = [Node() for _ in range(self.s)]
        self.build(0, self.n - 1, 1)

    def build(self, start, end, index):  # Never change this
        if start == end:
            self.tree[index] = self.Node(self.arr[start])
            return
        mid = (start + end) // 2
        self.build(start, mid, 2 * index)
        self.build(mid + 1, end, 2 * index + 1)
        self.tree[index] = self.Node()
        self.tree[index].merge(self.tree[2 * index], self.tree[2 * index + 1])

    def update(self, start, end, index, query_index, u):  # Never change this
        if start == end:
            u.apply(self.tree[index])
            return
        mid = (start + end) // 2
        if query_index <= mid:
            self.update(start, mid, 2 * index, query_index, u)
        else:
            self.update(mid + 1, end, 2 * index + 1, query_index, u)
        self.tree[index] = self.Node()
        self.tree[index].merge(self.tree[2 * index], self.tree[2 * index + 1])

    def query(self, start, end, index, left, right):  # Never change this
        if start > right or end < left:
            return self.Node()
        if start >= left and end <= right:
            return self.tree[index]
        mid = (start + end) // 2
        l = self.query(start, mid, 2 * index, left, right)
        r = self.query(mid + 1, end, 2 * index + 1, left, right)
        ans = self.Node()
        ans.merge(l, r)
        return ans

    def make_update(self, index, val):  # pass in as many parameters as required
        new_update = self.Update(val)  # may change
        self.update(0, self.n - 1, 1, index, new_update)

    def make_query(self, left, right):
        return self.query(0, self.n - 1, 1, left, right)

class Node1:
    def __init__(self, p1=0):  # Identity + actual node
        self.val = p1  # may change

    def merge(self, l, r):  # Merge two child nodes
        self.val = l.val ^ r.val  # may change


class Update1:
    def __init__(self, p1):  # Actual update
        self.val = p1  # may change

    def apply(self, a):  # apply update to given node
        a.val = self.val  # may change

# arr = [1, 2, 3, 4]
# seg = SegTree(len(arr), arr, Node1, Update1)
# # Query XOR from index 1 to 3
# print(seg.make_query(1, 3).val)  # Output: 2^3^4 = 5
# # Update index 2 to 10
# seg.make_update(2, 10)
# # Query again
# print(seg.make_query(1, 3).val)  # Output: 2^10^4 = 12

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
