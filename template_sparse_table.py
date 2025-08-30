from math import gcd, floor, log2

class SparseTable:
    ### Authored By: Akshit Singhvi
    def __init__(self, arr, func):
        self.func = func  # Operation: gcd, min, or max
        self.N = len(arr)
        self.K = floor(log2(self.N)) + 1
        self.st = [[0] * self.K for _ in range(self.N)]
        self.log = [0] * (self.N + 1)
        self._build_log()
        self._build_sparse_table(arr)

    def _build_log(self):
        for i in range(2, self.N + 1):
            self.log[i] = self.log[i // 2] + 1

    def _build_sparse_table(self, arr):
        for i in range(self.N):
            self.st[i][0] = arr[i]
        for j in range(1, self.K):
            for i in range(self.N - (1 << j) + 1):
                self.st[i][j] = self.func(
                    self.st[i][j - 1],
                    self.st[i + (1 << (j - 1))][j - 1]
                )

    def query(self, L, R):
        j = self.log[R - L + 1]
        return self.func(
            self.st[L][j],
            self.st[R - (1 << j) + 1][j]
        )
