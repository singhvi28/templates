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

def mobius(n):
    """Calculate Möbius function values up to n"""
    mu = [0] * (n + 1)
    mu[1] = 1
    is_prime = [True] * (n + 1)
    
    for i in range(2, n + 1):
        if is_prime[i]:
            for j in range(i, n + 1, i):
                is_prime[j] = False
                mu[j] = -1 if mu[j] == 0 else -mu[j]
            
            # Check for squares
            if i * i <= n:
                for j in range(i * i, n + 1, i * i):
                    mu[j] = 0
    
    # Correct calculation
    mu = [0] * (n + 1)
    is_prime = [True] * (n + 1)
    prime_factors = [0] * (n + 1)
    
    mu[1] = 1
    
    for i in range(2, n + 1):
        if is_prime[i]:
            for j in range(i, n + 1, i):
                is_prime[j] = False if j > i else True
                prime_factors[j] += 1
                
                # Check if divisible by i^2
                if j % (i * i) == 0:
                    prime_factors[j] = -1
    
    for i in range(1, n + 1):
        if prime_factors[i] == -1:
            mu[i] = 0
        elif prime_factors[i] == 0:
            mu[i] = 1
        else:
            mu[i] = -1 if prime_factors[i] % 2 == 1 else 1
    
    return mu


def mobius_sieve(n):
    """Optimized sieve for Möbius function"""
    mu = [1] * (n + 1)
    is_prime = [True] * (n + 1)
    primes = []
    
    for i in range(2, n + 1):
        if is_prime[i]:
            primes.append(i)
            mu[i] = -1
        
        for p in primes:
            if i * p > n:
                break
            is_prime[i * p] = False
            if i % p == 0:
                mu[i * p] = 0
                break
            mu[i * p] = -mu[i]
    
    return mu

