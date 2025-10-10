from typing import List

class Hashing:
    def __init__(self, s: str):
        self.s = s
        self.n = len(s)
        self.hashPrimes = [1000000009, 100000007]
        self.primes = len(self.hashPrimes)
        self.base = 31
        self.hashValues = [[] for _ in range(self.primes)]
        self.powersOfBase = [[] for _ in range(self.primes)]
        self.inversePowersOfBase = [[] for _ in range(self.primes)]

        for i in range(self.primes):
            mod = self.hashPrimes[i]
            self.powersOfBase[i] = [1] * (self.n + 1)
            self.inversePowersOfBase[i] = [1] * (self.n + 1)
            for j in range(1, self.n + 1):
                self.powersOfBase[i][j] = (self.powersOfBase[i][j - 1] * self.base) % mod
            self.inversePowersOfBase[i][self.n] = pow(self.powersOfBase[i][self.n], mod - 2, mod)
            for j in range(self.n - 1, -1, -1):
                self.inversePowersOfBase[i][j] = (self.inversePowersOfBase[i][j + 1] * self.base) % mod

        for i in range(self.primes):
            mod = self.hashPrimes[i]
            self.hashValues[i] = [0] * self.n
            for j in range(self.n):
                char_val = ord(s[j]) - ord('a') + 1
                hashed = (char_val * self.powersOfBase[i][j]) % mod
                if j > 0:
                    hashed = (hashed + self.hashValues[i][j - 1]) % mod
                self.hashValues[i][j] = hashed

    def substring_hash(self, l: int, r: int) -> List[int]:
        hash_result = [0] * self.primes
        for i in range(self.primes):
            mod = self.hashPrimes[i]
            val1 = self.hashValues[i][r]
            val2 = self.hashValues[i][l - 1] if l > 0 else 0
            sub_hash = (val1 - val2 + mod) % mod
            hash_result[i] = (sub_hash * self.inversePowersOfBase[i][l]) % mod
        return hash_result

def kmp_search(text, pattern):
    """
    Finds all starting indices where the pattern string occurs in the text string using the KMP algorithm.

    Args:
        text (str): The text string to search within.
        pattern (str): The pattern string to search for.

    Returns:
        List[int]: A list of starting indices where the pattern occurs in the text.
    
    Authored By:
        akkisinghvi28
    """
    if not pattern or not text:
        return []

    # Preprocess pattern to get LPS (Longest Prefix Suffix) array
    lps = [0] * len(pattern)
    length = 0
    for i in range(1, len(pattern)):
        while length > 0 and pattern[i] != pattern[length]:
            length = lps[length - 1]
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length

    # KMP search 
    result = []
    i = j = 0
    while i < len(text):
        if text[i] == pattern[j]:
            i += 1
            j += 1
            if j == len(pattern):
                result.append(i - j)
                j = lps[j - 1]
        else:
            if j > 0: j = lps[j - 1]
            else: i += 1

    return result


from collections import deque

class AhoCorasick:
    """
    Aho-Corasick automaton for multi-pattern string matching.

    Attributes:
        trie (List[Dict[char, int]]): The trie representing the automaton.
        fail (List[int]): Failure links for each node.
        output (List[Set[int]]): Set of pattern indices ending at each node.
        size (int): Total number of nodes in the trie.
    
    Authored By:
        akkisinghvi28
    """

    def __init__(self):
        """
        Initialises the automaton with an empty root node.
        """
        self.trie = [{}]
        self.fail = [0]
        self.output = [set()]
        self.size = 1

    def add(self, word, idx):
        """
        Inserts a pattern into the trie.

        Args:
            word (str): The pattern string to insert.
            idx (int): A unique identifier for the pattern (e.g. its index in the pattern list).
        """
        node = 0
        for c in word:
            if c not in self.trie[node]:
                self.trie.append({})
                self.fail.append(0)
                self.output.append(set())
                self.trie[node][c] = self.size
                self.size += 1
            node = self.trie[node][c]
        self.output[node].add(idx)

    def build(self):
        """
        Constructs the failure links for the trie using BFS.
        Must be called after all patterns have been added via `add()`.
        """
        q = deque()
        for c, nxt in self.trie[0].items():
            q.append(nxt)
            self.fail[nxt] = 0

        while q:
            node = q.popleft()
            for c, nxt in self.trie[node].items():
                f = self.fail[node]
                while f and c not in self.trie[f]:
                    f = self.fail[f]
                self.fail[nxt] = self.trie[f][c] if c in self.trie[f] else 0
                self.output[nxt] |= self.output[self.fail[nxt]]
                q.append(nxt)

    def search(self, text):
        """
        Searches for all occurrences of the added patterns in the given text.

        Args:
            text (str): The text in which to search.

        Returns:
            List[Tuple[int, int]]: A list of (end_index, pattern_index) pairs indicating
            that the pattern with pattern_index ends at end_index in the text.
        """
        node = 0
        results = []
        for i, c in enumerate(text):
            while node and c not in self.trie[node]:
                node = self.fail[node]
            if c in self.trie[node]:
                node = self.trie[node][c]
            else:
                node = 0
            for pat_idx in self.output[node]:
                results.append((i, pat_idx))
        return results


def manacher(s):
    """
    Returns (length, start, end) of the longest palindromic substring in s.
    """
    t = "#".join("^" + s + "$")
    n = len(t)
    p = [0] * n
    c = r = 0

    for i in range(1, n - 1):
        mirror = 2 * c - i
        if i < r:
            p[i] = min(r - i, p[mirror])
        while t[i + 1 + p[i]] == t[i - 1 - p[i]]:
            p[i] += 1
        if i + p[i] > r:
            c, r = i, i + p[i]

    max_len, center = max((v, i) for i, v in enumerate(p))
    start = (center - max_len) // 2
    end = start + max_len - 1
    return max_len, start, end
