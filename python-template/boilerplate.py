#         __                __          __________  ______  
#   _____|__| ____    ____ |  |_____  _|__\_____  \/  __  \ 
#  /  ___|  |/    \  / ___\|  |  \  \/ /  |/  ____/\      / 
#  \___ \|  |   |  \/ /_/  >   Y  \   /|  /       \/  --  \
#  /_____|__|___|  /\___  /|___|  /\_/ |__\_______/\______/
#                \//_____/      \/                         

import sys
DEBUG = False
# Only redirect when running locally
if sys.stdin.isatty():
    try:
        DEBUG = True
        sys.stdin = open('input.txt', 'r')
        sys.stdout = open('output.txt', 'w')
    except:
        pass

from sys import stdin
inpInt = lambda: int(stdin.readline())
inpMap = lambda: map(int, stdin.readline().split())
inpList = lambda: list(map(int, stdin.readline().split()))
inpFloat = lambda: map(float, stdin.readline().split())
inpStr = lambda: stdin.readline().rstrip('\n')
flush = lambda: sys.stdout.flush()
from itertools import accumulate
prefixSum = lambda a: list(accumulate(a))
suffixSum = lambda a: list(accumulate(a[::-1]))[::-1]
prefixMax = lambda a: list(accumulate(a, max))
suffixMax = lambda a: list(accumulate(a[::-1], max))[::-1]
prefixMin = lambda a: list(accumulate(a, min))
suffixMin = lambda a: list(accumulate(a[::-1], min))[::-1]
from collections import deque
from heapq import heapify, heappop, heappush
from math import gcd, ceil, sqrt, isqrt, log2
from fractions import Fraction
from typing import List, Tuple
from bisect import bisect_left, bisect_right, insort
from functools import reduce, lru_cache
from random import randrange
# sys.setrecursionlimit(5*(10**5))

# For writing recursions
# Use yield instead of return
from types import GeneratorType
def bootstrap(f, stack=[]):
    def wrappedfunc(*args, **kwargs):
        if stack:
            return f(*args, **kwargs)
        else:
            to = f(*args, **kwargs)
            while True:
                if type(to) is GeneratorType:
                    stack.append(to)
                    to = next(to)
                else:
                    stack.pop()
                    if not stack:
                        break
                    to = stack[-1].send(to)
            return to
    return wrappedfunc

class SafeMap:
    __slots__ = ("_d", "_seed")

    def __init__(self):
        self._d = {}
        self._seed = randrange(1 << 61)

    def _hash(self, x):
        # splitmix64
        x += self._seed
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9
        x &= (1 << 64) - 1
        x = (x ^ (x >> 27)) * 0x94d049bb133111eb
        x &= (1 << 64) - 1
        return x ^ (x >> 31)

    def __setitem__(self, key, value):
        self._d[self._hash(key)] = value

    def __getitem__(self, key):
        return self._d[self._hash(key)]

    def __contains__(self, key):
        return self._hash(key) in self._d

    def get(self, key, default=None):
        return self._d.get(self._hash(key), default)

    def pop(self, key, default=None):
        return self._d.pop(self._hash(key), default)

    def clear(self):
        self._d.clear()

    def items(self):
        return self._d.items()

    def values(self):
        return self._d.values()

    def keys(self):
        return self._d.keys()

    def __len__(self):
        return len(self._d)

class Counter(SafeMap):
    __slots__ = ()

    def __init__(self, iterable=None):
        super().__init__()
        if iterable:
            for x in iterable:
                self[x] += 1

    def __getitem__(self, key):
        return self._d.get(self._hash(key), 0)

    def increment(self, key, value=1):
        h = self._hash(key)
        self._d[h] = self._d.get(h, 0) + value

    def decrement(self, key, value=1):
        h = self._hash(key)
        self._d[h] = self._d.get(h, 0) - value

class SafeSet:
    __slots__ = ("_d", "_seed")

    def __init__(self, iterable=None):
        self._d = {}
        self._seed = randrange(1 << 61)
        if iterable:
            for item in iterable:
                self.add(item)

    def _hash(self, x):
        # splitmix64 logic to prevent hash collisions
        x += self._seed
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9
        x &= (1 << 64) - 1
        x = (x ^ (x >> 27)) * 0x94d049bb133111eb
        x &= (1 << 64) - 1
        return x ^ (x >> 31)

    def add(self, item):
        """Add an element to the set."""
        self._d[self._hash(item)] = item

    def discard(self, item):
        """Remove an element from a set if it is a member. If not, do nothing."""
        self._d.pop(self._hash(item), None)

    def remove(self, item):
        """Remove an element; raises KeyError if the element is not present."""
        hashed = self._hash(item)
        if hashed in self._d:
            del self._d[hashed]
        else:
            raise KeyError(item)

    def __contains__(self, item):
        return self._hash(item) in self._d

    def __len__(self):
        return len(self._d)

    def __iter__(self):
        # We yield the original values, not the hashes
        return iter(self._d.values())

    def clear(self):
        self._d.clear()

    def pop(self):
        """Remove and return an arbitrary set element."""
        if not self._d:
            raise KeyError('pop from an empty set')
        # Popitem returns a (hash, value) tuple
        _, val = self._d.popitem()
        return val

    def __repr__(self):
        return f"SafeSet({list(self._d.values())})"

class defaultdict:
    __slots__ = ("_d", "_seed", "_factory")

    def __init__(self, default_factory):
        self._d = {}
        self._factory = default_factory
        self._seed = id(self) ^ 0x9e3779b97f4a7c15

    def _hash(self, x):
        x = (x ^ self._seed) & ((1 << 64) - 1)
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9 & ((1 << 64) - 1)
        x = (x ^ (x >> 27)) * 0x94d049bb133111eb & ((1 << 64) - 1)
        return x ^ (x >> 31)

    def __getitem__(self, key):
        h = self._hash(key)
        bucket = self._d.setdefault(h, [])

        for k, v in bucket:
            if k == key:
                return v

        v = self._factory()
        bucket.append([key, v])
        return v

    def __setitem__(self, key, value):
        h = self._hash(key)
        bucket = self._d.setdefault(h, [])

        for pair in bucket:
            if pair[0] == key:
                pair[1] = value
                return

        bucket.append([key, value])

    def keys(self):
        for bucket in self._d.values():
            for k, _ in bucket:
                yield k

    def values(self):
        for bucket in self._d.values():
            for _, v in bucket:
                yield v

    def items(self):
        for bucket in self._d.values():
            for k, v in bucket:
                yield (k, v)

INF = float('inf')
MOD = 10**9 + 7
# MOD = 10**9 + 9
# MOD = 10**9 + 21
# MOD = 676767677
# MOD = 998244353

def MODADD(*args):
    res = 0
    for num in args:
        res += num
        res %= MOD
    return res

def MODMUL(*args):
    result = 1
    for num in args:
        result = (result * num) % MOD
    return result

def pyn(b):
    if b: print("Yes")
    else: print("No")

##### CODE STARTS HERE

def solve():
    pass

def main():
    test = 1
    # if DEBUG: test = inpInt() # for AtCoder
    # test = inpInt() # for Codeforces
    for _ in range(test):
        # print(solve())
        solve()
if __name__ == "__main__":
    main()
