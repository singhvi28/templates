# BIT MANIPULATION TRICKS
# a | b = a ^ b + a & b
# a ^ (a & b) = (a | b) ^ b
# b ^ (a & b) = (a | b) ^ a
# (a & b) ^ (a | b) = a ^ b
# a + b = a | b + a & b
# a + b = a ^ b + 2 * (a & b)
# a - b = (a ^ (a & b)) - ((a | b) ^ a)
# a - b = ((a | b) ^ b) - ((a | b) ^ a)
# a - b = (a ^ (a & b)) - (b ^ (a & b))
# a - b = ((a | b) ^ b) - (b ^ (a & b))
# if x is a power of two ---> x&(x-1) == 0

### GAUSS ELIMINATION

def gauss_solve(A, b):
    """
    Solve A x = b using Gaussian elimination.
    
    Input:
        A : list of lists (n × m matrix)
        b : list (n vector)

    Returns:
        (status, solution)
            status = 1  -> unique solution
            status = 2  -> infinite solutions
            status = -1 -> no solution
            solution = list of variable values if exist
    
    Author: akshitsinghvi28
    """

    a = [row + [b[i]] for i, row in enumerate(A)]

    eps = 1e-9
    n = len(a)
    m = len(a[0]) - 1
    pos = [-1] * m
    rank = 0

    for col in range(m):
        row = rank
        mx = row

        for i in range(row, n):
            if abs(a[i][col]) > abs(a[mx][col]):
                mx = i

        if abs(a[mx][col]) < eps:
            continue

        a[row], a[mx] = a[mx], a[row]
        pos[col] = row

        for i in range(n):
            if i != row and abs(a[i][col]) > eps:
                c = a[i][col] / a[row][col]
                for j in range(col, m+1):
                    a[i][j] -= a[row][j] * c

        rank += 1

    ans = [0] * m
    for i in range(m):
        if pos[i] != -1:
            ans[i] = a[pos[i]][m] / a[pos[i]][i]

    for i in range(n):
        s = sum(ans[j] * a[i][j] for j in range(m))
        if abs(s - a[i][m]) > eps:
            return -1, []

    for i in range(m):
        if pos[i] == -1:
            return 2, ans

    return 1, ans


### MATRIX EXPONENTIATION
from typing import List

L = 26
MOD = 10**9 + 7

class Mat:
    def __init__(self, copy_from: "Mat" = None) -> None:
        self.a: List[List[int]] = [[0] * L for _ in range(L)]
        if copy_from:
            for i in range(L):
                for j in range(L):
                    self.a[i][j] = copy_from.a[i][j]

    def __mul__(self, other: "Mat") -> "Mat":
        result = Mat()
        for i in range(L):
            for j in range(L):
                for k in range(L):
                    result.a[i][j] = (
                        result.a[i][j] + self.a[i][k] * other.a[k][j]
                    ) % MOD
        return result

    def transpose(self) -> "Mat":
        """Return the transpose of the matrix as a new Mat object."""
        result = Mat()
        for i in range(L):
            for j in range(L):
                result.a[j][i] = self.a[i][j]
        return result


def I() -> Mat:
    m = Mat()
    for i in range(L):
        m.a[i][i] = 1
    return m

def quickmul(x: Mat, y: int) -> Mat:
    ans = I()
    cur = x
    while y:
        if y & 1:
            ans = ans * cur
        cur = cur * cur
        y >>= 1
    return ans



class Combinatorics:
    """
    Efficient nCr (binomial coefficient) modulo MOD with precomputation.

    Precomputes factorials up to N, so each nCr query is O(1).
    Assumes MOD is prime (default = 1e9+7).
    """

    def __init__(self, N: int, 
            MOD: int = 10**9 + 7
            # MOD: int = 998244353    
        ):
        self.N = N
        self.MOD = MOD
        self.fact = [1] * (N + 1)
        for i in range(1, N + 1):
            self.fact[i] = (self.fact[i - 1] * i) % self.MOD
        self.d = self._get_derangements(N)
    
    def _inversemod(self, a: int) -> int:
        """Return modular inverse of a modulo MOD (MOD must be prime)."""
        return pow(a, self.MOD - 2, self.MOD)

    def nCr(self, n: int, r: int) -> int:
        """Return nCr % MOD, or 0 if r > n."""
        if r > n: return 0
        p1 = (self.fact[n] * self._inversemod(self.fact[r])) % self.MOD
        p2 = self._inversemod(self.fact[n - r])
        return (p1 * p2) % self.MOD

    # Function to calculate derangements !n
    # !n=(n−1)(!(n−1)+!(n−2)) where !0 = 1 and !1 = 0
    # Time complexity: O(n)
    def _get_derangements(self, n):
        """Return derangements upto n."""
        if n == 0: return 1
        if n == 1: return 0
        dp = [0] * (n + 1)
        dp[0] = 1
        dp[1] = 0
        for i in range(2, n + 1): dp[i] = ((i - 1) * (dp[i - 1] + dp[i - 2])) % MOD
        return dp

    def derangement(self, n): 
        return self.d[n]

class ExactCombinatorics:
    def __init__(self, N: int):
        self.N = N
        self.fact = [1] * (N + 1)
        for i in range(1, N + 1):
            self.fact[i] = self.fact[i - 1] * i
        self.d = self._get_derangements(N)

    def nCr(self, n: int, r: int) -> int:
        """Compute exact n choose r."""
        if r > n or n < 0 or r < 0:
            return 0
        return self.fact[n] // (self.fact[r] * self.fact[n - r])

    def _get_derangements(self, n: int):
        """Precompute derangements exactly (no mod)."""
        if n == 0:
            return [1]
        if n == 1:
            return [1, 0]
        dp = [0] * (n + 1)
        dp[0] = 1
        dp[1] = 0
        for i in range(2, n + 1):
            dp[i] = (i - 1) * (dp[i - 1] + dp[i - 2])
        return dp

    def derangement(self, n: int) -> int:
        """Return exact derangement count for n."""
        if n > self.N:
            raise ValueError(f"Precomputed only up to {self.N}")
        return self.d[n]
