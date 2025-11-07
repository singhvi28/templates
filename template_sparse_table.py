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


from math import gcd, log2, floor

class SparseTable2D:
    ### Authored By: Akshit Singhvi
    def __init__(self, matrix, func):
        self.func = func  # Operation: gcd, min, or max
        self.N = len(matrix)
        self.M = len(matrix[0])
        self.K1 = floor(log2(self.N)) + 1
        self.K2 = floor(log2(self.M)) + 1

        self.logN = [0]*(self.N+1)
        self.logM = [0]*(self.M+1)
        for i in range(2, self.N+1):
            self.logN[i] = self.logN[i//2] + 1
        for j in range(2, self.M+1):
            self.logM[j] = self.logM[j//2] + 1

        self.st = [[[[0]*self.K2 for _ in range(self.K1)]
                       for _ in range(self.M)]
                       for _ in range(self.N)]

        self._build(matrix)

    def _build(self, matrix):
        for i in range(self.N):
            for j in range(self.M):
                self.st[i][j][0][0] = matrix[i][j]

        for k2 in range(1, self.K2):
            step = 1 << (k2 - 1)
            for i in range(self.N):
                for j in range(self.M - (1 << k2) + 1):
                    self.st[i][j][0][k2] = self.func(
                        self.st[i][j][0][k2-1],
                        self.st[i][j + step][0][k2-1]
                    )

        for k1 in range(1, self.K1):
            step = 1 << (k1 - 1)
            for i in range(self.N - (1 << k1) + 1):
                for j in range(self.M):
                    for k2 in range(self.K2):
                        self.st[i][j][k1][k2] = self.func(
                            self.st[i][j][k1-1][k2],
                            self.st[i + step][j][k1-1][k2]
                        )

    def query(self, x1, y1, x2, y2):
        k1 = self.logN[x2 - x1 + 1]
        k2 = self.logM[y2 - y1 + 1]
        x_step = 1 << k1
        y_step = 1 << k2
        return self.func(
            self.func(
                self.st[x1][y1][k1][k2],
                self.st[x2 - x_step + 1][y1][k1][k2]
            ),
            self.func(
                self.st[x1][y2 - y_step + 1][k1][k2],
                self.st[x2 - x_step + 1][y2 - y_step + 1][k1][k2]
            )
        )
