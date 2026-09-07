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

class LazySegTree:
    """
    Iterative Lazy Segment Tree supporting:
    1. Range updates: apply(l, r, f) or point update: apply(p, f)
    2. Range queries: prod(l, r) or all_prod()
    3. Point access: get(p), set(p, val)
    4. Tree walks: find_first(l, r, pred), find_last(l, r, pred)
                   max_right(l, g), min_left(r, g)
    """

    def __init__(self, arr, op, e, mapping, composition, id_lazy):
        if isinstance(arr, int):
            self.n = arr
            arr = [e] * arr
        else:
            self.n = len(arr)
        self.op = op
        self.e = e
        self.mapping = mapping
        self.composition = composition
        self.id_lazy = id_lazy

        self.size = 1 << (self.n - 1).bit_length() if self.n > 0 else 1
        self.log = self.size.bit_length() - 1
        self.tree = [e] * (2 * self.size)
        self.lazy = [id_lazy] * self.size

        for i in range(self.n):
            self.tree[self.size + i] = arr[i]
        for i in range(self.size - 1, 0, -1):
            self._update(i)

    def _update(self, node):
        self.tree[node] = self.op(self.tree[2 * node], self.tree[2 * node + 1])

    def _apply_node(self, node, f):
        self.tree[node] = self.mapping(f, self.tree[node])
        if node < self.size:
            self.lazy[node] = self.composition(f, self.lazy[node])

    def _push(self, node):
        if self.lazy[node] != self.id_lazy:
            f = self.lazy[node]
            self._apply_node(2 * node, f)
            self._apply_node(2 * node + 1, f)
            self.lazy[node] = self.id_lazy

    def apply(self, l, r, f=None):
        """Apply lazy tag f over range [l, r). If f is None, point update: apply(p, f)."""
        if f is None:
            p, f = l, r
            p += self.size
            for i in range(self.log, 0, -1):
                self._push(p >> i)
            self._apply_node(p, f)
            for i in range(1, self.log + 1):
                self._update(p >> i)
            return

        if l >= r or l >= self.n or r <= 0: return
        l = max(l, 0)
        r = min(r, self.n)
        l += self.size
        r += self.size

        for i in range(self.log, 0, -1):
            if ((l >> i) << i) != l: self._push(l >> i)
            if ((r >> i) << i) != r: self._push((r - 1) >> i)

        l2, r2 = l, r
        while l < r:
            if l & 1:
                self._apply_node(l, f)
                l += 1
            if r & 1:
                r -= 1
                self._apply_node(r, f)
            l >>= 1
            r >>= 1

        l, r = l2, r2
        for i in range(1, self.log + 1):
            if ((l >> i) << i) != l: self._update(l >> i)
            if ((r >> i) << i) != r: self._update((r - 1) >> i)

    def prod(self, l, r):
        """Query aggregate over range [l, r)."""
        if l >= r or l >= self.n or r <= 0: return self.e
        l = max(l, 0)
        r = min(r, self.n)
        l += self.size
        r += self.size

        for i in range(self.log, 0, -1):
            if ((l >> i) << i) != l: self._push(l >> i)
            if ((r >> i) << i) != r: self._push((r - 1) >> i)

        sml = self.e
        smr = self.e
        while l < r:
            if l & 1:
                sml = self.op(sml, self.tree[l])
                l += 1
            if r & 1:
                r -= 1
                smr = self.op(self.tree[r], smr)
            l >>= 1
            r >>= 1
        return self.op(sml, smr)

    def all_prod(self):
        """Return the aggregate of the whole array."""
        return self.tree[1]

    def get(self, p):
        """Return value at index p."""
        p += self.size
        for i in range(self.log, 0, -1):
            self._push(p >> i)
        return self.tree[p]

    def set(self, p, val):
        """Set arr[p] = val."""
        p += self.size
        for i in range(self.log, 0, -1):
            self._push(p >> i)
        self.tree[p] = val
        for i in range(1, self.log + 1):
            self._update(p >> i)

    def find_first(self, l, r, pred):
        """Find leftmost index in [l, r) satisfying pred(tree[node])."""
        if l >= r or l >= self.n or r <= 0: return -1
        l = max(l, 0)
        r = min(r, self.n)
        l += self.size
        r += self.size

        for i in range(self.log, 0, -1):
            if ((l >> i) << i) != l: self._push(l >> i)
            if ((r >> i) << i) != r: self._push((r - 1) >> i)

        left_nodes = []
        right_nodes = []
        while l < r:
            if l & 1:
                left_nodes.append(l)
                l += 1
            if r & 1:
                r -= 1
                right_nodes.append(r)
            l >>= 1
            r >>= 1

        for node in left_nodes + right_nodes[::-1]:
            if pred(self.tree[node]):
                while node < self.size:
                    self._push(node)
                    if pred(self.tree[2 * node]):
                        node = 2 * node
                    else:
                        node = 2 * node + 1
                return node - self.size
        return -1

    def find_last(self, l, r, pred):
        """Find rightmost index in [l, r) satisfying pred(tree[node])."""
        if l >= r or l >= self.n or r <= 0: return -1
        l = max(l, 0)
        r = min(r, self.n)
        l += self.size
        r += self.size

        for i in range(self.log, 0, -1):
            if ((l >> i) << i) != l: self._push(l >> i)
            if ((r >> i) << i) != r: self._push((r - 1) >> i)

        left_nodes = []
        right_nodes = []
        while l < r:
            if l & 1:
                left_nodes.append(l)
                l += 1
            if r & 1:
                r -= 1
                right_nodes.append(r)
            l >>= 1
            r >>= 1

        for node in right_nodes + left_nodes[::-1]:
            if pred(self.tree[node]):
                while node < self.size:
                    self._push(node)
                    if pred(self.tree[2 * node + 1]):
                        node = 2 * node + 1
                    else:
                        node = 2 * node
                return node - self.size
        return -1

    def max_right(self, l, g):
        """ACL max_right: find largest r in [l, n] s.t. g(op(arr[l], ..., arr[r-1])) is True."""
        if l >= self.n: return self.n
        l = max(l, 0)
        l += self.size
        for i in range(self.log, 0, -1):
            self._push(l >> i)
        sm = self.e
        while True:
            while l % 2 == 0: l >>= 1
            if not g(self.op(sm, self.tree[l])):
                while l < self.size:
                    self._push(l)
                    l = 2 * l
                    if g(self.op(sm, self.tree[l])):
                        sm = self.op(sm, self.tree[l])
                        l += 1
                return l - self.size
            sm = self.op(sm, self.tree[l])
            l += 1
            if (l & -l) == l: break
        return self.n

    def min_left(self, r, g):
        """ACL min_left: find smallest l in [0, r] s.t. g(op(arr[l], ..., arr[r-1])) is True."""
        if r <= 0: return 0
        r = min(r, self.n)
        r += self.size
        for i in range(self.log, 0, -1):
            self._push((r - 1) >> i)
        sm = self.e
        while True:
            r -= 1
            while r > 1 and (r % 2): r >>= 1
            if not g(self.op(self.tree[r], sm)):
                while r < self.size:
                    self._push(r)
                    r = 2 * r + 1
                    if g(self.op(self.tree[r], sm)):
                        sm = self.op(self.tree[r], sm)
                        r -= 1
                return r + 1 - self.size
            sm = self.op(self.tree[r], sm)
            if (r & -r) == r: break
        return 0


import bisect

class MergeSortTreeFractionalCascading:
    def __init__(self, arr):
        """
        Iteratively builds a Merge Sort Tree enhanced with Fractional Cascading.
        Time Complexity: O(N log N)
        Space Complexity: O(N log N)
        """
        self.n = len(arr)
        if self.n == 0:
            return
            
        # Pad N to the next power of 2 for clean top-down binary splits
        self.N = 1
        while self.N < self.n:
            self.N *= 2
            
        # tree stores the sorted arrays for each node
        self.tree = [[] for _ in range(2 * self.N)]
        
        # to_left[node][k] stores how many of the first 'k' elements 
        # in tree[node] originated from its left child.
        self.to_left = [[] for _ in range(self.N)]
        
        # 1. Initialize leaves
        for i in range(self.N):
            if i < self.n:
                self.tree[self.N + i] = [arr[i]]
            else:
                self.tree[self.N + i] = [float('inf')] # Pad out-of-bounds with infinity
                
        # 2. Build internal nodes iteratively (bottom-up)
        for i in range(self.N - 1, 0, -1):
            left_child = 2 * i
            right_child = 2 * i + 1
            
            L_arr = self.tree[left_child]
            R_arr = self.tree[right_child]
            
            len_L = len(L_arr)
            len_R = len(R_arr)
            
            merged = []
            to_left_arr = [0]
            
            p1 = 0
            p2 = 0
            
            # Merge left and right sorted arrays while mapping fractional cascading pointers
            while p1 < len_L and p2 < len_R:
                if L_arr[p1] <= R_arr[p2]:
                    merged.append(L_arr[p1])
                    p1 += 1
                else:
                    merged.append(R_arr[p2])
                    p2 += 1
                to_left_arr.append(p1)
                
            # Exhaust remaining elements
            while p1 < len_L:
                merged.append(L_arr[p1])
                p1 += 1
                to_left_arr.append(p1)
                
            while p2 < len_R:
                merged.append(R_arr[p2])
                p2 += 1
                to_left_arr.append(p1) # p1 remains constant
                
            self.tree[i] = merged
            self.to_left[i] = to_left_arr

    def count_less_equal(self, L, R, x):
        """
        Iteratively counts the number of elements <= x in the original array's range [L, R].
        Time Complexity: O(log N)
        """
        if L > R or L < 0 or R >= self.n:
            return 0
            
        # O(log N) - The ONLY binary search happens at the root!
        # k represents the number of elements <= x in the entire segment.
        k = bisect.bisect_right(self.tree[1], x)
        if k == 0:
            return 0
            
        ans = 0
        # Stack simulates a top-down segment tree traversal iteratively.
        # Stores: (node_idx, node_range_L, node_range_R, query_range_L, query_range_R, current_k)
        stack = [(1, 0, self.N - 1, L, R, k)]
        
        while stack:
            node, nl, nr, ql, qr, cur_k = stack.pop()
            
            # If the current node's range is completely inside the query range
            if ql <= nl and nr <= qr:
                ans += cur_k
                continue
                
            mid = (nl + nr) // 2
            
            # Fractional Cascading Magic: O(1) transitions to children bounds!
            k_left = self.to_left[node][cur_k]
            k_right = cur_k - k_left
            
            # If the left child's range overlaps with our query
            if ql <= mid and k_left > 0:
                stack.append((node * 2, nl, mid, ql, min(qr, mid), k_left))
                
            # If the right child's range overlaps with our query
            if qr > mid and k_right > 0:
                stack.append((node * 2 + 1, mid + 1, nr, max(ql, mid + 1), qr, k_right))
                
        return ans

    def count_in_range(self, L, R, x, y):
        """
        Counts the number of elements in the original array's range [L, R] that are in the value range [x, y].
        Time Complexity: O(log N)
        """
        if x > y:
            return 0
        return self.count_less_equal(L, R, y) - self.count_less_equal(L, R, x - 1)