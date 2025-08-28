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

MOD = 10**9 + 7
NUM = 10^7 + 30
N = 10^7 + 30

# Function to initialize power array
# Time complexity: O(N)
def initpow(x):
    power = [1] * NUM
    for i in range(1, NUM):
        power[i] = (power[i-1] * (x % MOD)) % MOD
    return power 

# Function to calculate sieve of Eratosthenes
# Time complexity: O(N log log N)
# def calc_sieve(LIMIT):
#     sieve = [0] * (LIMIT + 1)
#     for x in range(2, LIMIT + 1):
#         if sieve[x] == 0:  # x is prime
#             for u in range(x, LIMIT + 1, x):
#                 sieve[u] = x
#     return sieve

# def generate_primes(LIMIT):
#     sieve = calc_sieve(LIMIT) 
#     primes = [x for x in range(2, LIMIT + 1) if sieve[x] == x]
#     return primes

# Function to generate primes using Sieve of Euler
# Time complexity: O(n)

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
def primefactor(N):
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
    factors = []
    while n > 1:
        p = lp[n]
        count = 0
        while n % p == 0:
            n //= p
            count += 1
        factors.append((p, count))
    return factors

# Example usage
# N = 10**6  # You can set N according to your need
# lp, pr = primefactor(N)
# n = 360
# factors = get_prime_factors(n, lp)
# print(factors)

# Binary exponentiation function
# Time complexity: O(log b)
def binpow(a, b):
    res = 1
    while b > 0:
        if b & 1:
            res = res * a
        a = a * a
        b >>= 1
    return res

# Binary exponentiation function with modulus
# Time complexity: O(log b)
def binpow_mod(a, b, mod):
    res = 1
    while b > 0:
        if b & 1:
            res = (res * a) % mod
        a = (a * a) % mod
        b >>= 1
    return res % mod

# Function to calculate GCD
# Time complexity: O(log min(a, b))
# def gcd(a, b):
#     while b:
#         a, b = b, a % b
#     return a

# Function to check if number is prime
# Time complexity: O(n**0.5)
def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n ** 0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

# Function to calculate factors of number
# Time complexity: O(n**0.5)
def print_factors(n):
    for i in range(1, int(n ** 0.5) + 1):
        if n % i == 0:
            print(i)
            if i != n // i:
                print(n // i)

# Function to calculate LCM
# Time complexity: O(log min(a, b))
from math import gcd
def lcm(a, b):
    return (a // gcd(a, b)) * b

# Function to calculate modular inverse
# Time complexity: O(log mod)
def inversemod(a, mod):
    return binpow_mod(a, mod - 2, mod)

# Function for modular division
# Time complexity: O(log c)
def divmod(a, b, c):
    return (a % c * inversemod(b, c)) % c

# Function to calculate combination nCk modulo MOD
# Time complexity: O(1) after precomputing factorial array
def combination(n, k, fact):
    if k > n:
        return 0
    p1 = (fact[n] * inversemod(fact[k], MOD)) % MOD
    p2 = (1 * inversemod(fact[n - k], MOD)) % MOD
    return (p1 * p2) % MOD

# Function to calculate derangements !n
# !n=(n−1)(!(n−1)+!(n−2)) where !0 = 1 and !1 = 0
# Time complexity: O(n)
def derangement(n):
    if n == 0: return 1
    if n == 1: return 0
    dp = [0] * (n + 1)
    dp[0] = 1
    dp[1] = 0
    for i in range(2, n + 1): dp[i] = ((i - 1) * (dp[i - 1] + dp[i - 2])) % MOD
    return dp[n]

from typing import List
L = 26
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