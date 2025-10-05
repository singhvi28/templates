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



##### NUMBER THEORY

# Function to generate primes using Sieve of Euler
# Time complexity: O(n)
# Authored By; akkisinghvi28
def euler_sieve(n):
    is_prime = [True] * (n + 1)
    primes = []

    is_prime[0] = is_prime[1] = False

    for i in range(2, n + 1):
        if is_prime[i]:
            primes.append(i)
        for p in primes:
            if i * p > n:
                break
            is_prime[i * p] = False
            if i % p == 0:
                break

    return primes

# Function to calculate the prime factorization
# Time complexity: O(N log log N)
# Authored By; akkisinghvi28
def primefactor(N):
    """
    Linear sieve to compute smallest prime factors (SPF) up to N.

    Returns
    -------
    lp : list[int]  Smallest prime factor for each number.
    pr : list[int]  List of primes up to N.
    """
    lp = [0] * (N + 1)
    pr = []
    for i in range(2, N + 1):
        if lp[i] == 0:
            lp[i] = i
            pr.append(i)
        for j in range(len(pr)):
            if pr[j] > lp[i] or i * pr[j] > N:
                break
            lp[i * pr[j]] = pr[j]
    return lp, pr


def get_prime_factors(n, lp):
    """
    Factorises n using precomputed smallest prime factors.

    Returns
    -------
    list[tuple[int, int]] : Prime factors with exponents.
    """
    factors = []
    while n > 1:
        p = lp[n]
        count = 0
        while n % p == 0:
            n //= p
            count += 1
        factors.append((p, count))
    return factors

# # Example usage
# N = 10**6  # You can set N according to your need
# lp, pr = primefactor(N)
# n = 360
# factors = get_prime_factors(n, lp) # [(2, 3), (3, 2), (5, 1)]

# Function to calculate factors of number
# Time complexity: O(n**0.5)
def getFactors(n):
    res = []
    for i in range(1, int(n ** 0.5) + 1):
        if n % i == 0:
            res.append(i)
            if i != n // i: res.append(n // i)
    res.sort()
    return res

def egcd_iterative(a: int, b: int):
    """
    Extended Euclidean algorithm (iterative).
    Returns (g, x, y) where g = gcd(a, b) and a*x + b*y = g.
    TC = O(log(min(a,b))) | SC = O(1)
    """
    A, B = a, b
    x0, y0 = 1, 0
    x1, y1 = 0, 1
    while B != 0:
        q = A // B
        A, B = B, A - q * B
        x0, x1 = x1, x0 - q * x1
        y0, y1 = y1, y0 - q * y1
    if A < 0:
        A = -A
        x0 = -x0
        y0 = -y0
    return (A, x0, y0)

# BEZOUT'S LEMMA -> a*x + b*y = g, where g = gcd(a, b)

from typing import List

def phi(n: int, primes: List[int]) -> int:
    """
    Compute Euler's Totient function φ(n) using precomputed primes.
    TC = O(n**0.5 / log n) | SC = O(1)

    Parameters
    ----------
    n : int
        Input integer (n >= 1).
    primes : list[int]
        List of primes up to at least sqrt(n).

    Returns
    -------
    int
        Number of integers in [1..n] that are coprime to n.
    """
    result = n
    temp = n
    for p in primes:
        if p * p > temp:
            break
        if temp % p == 0:
            while temp % p == 0:
                temp //= p
            result -= result // p
    if temp > 1: result -= result // temp
    return result

# EULER'S THEOREM
# if gcd(a, n) = 1, then
# (a ** phi(n)) % n = 1

# FERMAT'S LITTLE THEOREM
# specific case of Euler's Theorem 
# where n = p (prime)
# a ** (p-1) % p = 1

# EXTENDED EULER'S THEOREM
# (works even when a and n are not coprime)
# (a ** b) % n  =  (a ** (phi(n) + b % phi(n))) % n
# provided b >= phi(n)

from typing import List, Tuple
def crt(remainders: List[int], moduli: List[int]) -> Tuple[int, int]:
    """
    Chinese Remainder Theorem solver for pairwise coprime moduli.
    Returns (x, M) where x is the solution modulo M = product(moduli).
    """
    assert len(remainders) == len(moduli)
    M = 1
    for m in moduli:
        M *= m
    
    x = 0
    for ai, mi in zip(remainders, moduli):
        Mi = M // mi
        # Modular inverse using pow (since mi is prime-safe)
        yi = pow(Mi, -1, mi)
        x = (x + ai * Mi * yi) % M
    return x, M

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