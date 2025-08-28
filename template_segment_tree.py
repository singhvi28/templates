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