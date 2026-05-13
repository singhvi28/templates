#include <bits/stdc++.h>
using namespace std;

using ll = long long;

ll mod_pow(ll a, ll e, ll mod) {
    ll res = 1 % mod;
    a %= mod;
    while (e) {
        if (e & 1) res = (__int128)res * a % mod;
        a = (__int128)a * a % mod;
        e >>= 1;
    }
    return res;
}

struct Hashing {
    string s;
    int n;
    vector<ll> hashPrimes = {1000000009LL, 100000007LL};
    int base = 31;
    vector<vector<ll>> hashValues, powersOfBase, inversePowersOfBase;

    Hashing(const string& s_) : s(s_), n((int)s_.size()) {
        int primes = (int)hashPrimes.size();
        hashValues.assign(primes, {});
        powersOfBase.assign(primes, {});
        inversePowersOfBase.assign(primes, {});

        for (int i = 0; i < primes; i++) {
            ll mod = hashPrimes[i];
            powersOfBase[i].assign(n + 1, 1);
            inversePowersOfBase[i].assign(n + 1, 1);
            for (int j = 1; j <= n; j++) powersOfBase[i][j] = powersOfBase[i][j - 1] * base % mod;
            inversePowersOfBase[i][n] = mod_pow(powersOfBase[i][n], mod - 2, mod);
            for (int j = n - 1; j >= 0; j--) inversePowersOfBase[i][j] = inversePowersOfBase[i][j + 1] * base % mod;
        }

        for (int i = 0; i < primes; i++) {
            ll mod = hashPrimes[i];
            hashValues[i].assign(n, 0);
            for (int j = 0; j < n; j++) {
                ll char_val = s[j] - 'a' + 1;
                ll hashed = char_val * powersOfBase[i][j] % mod;
                if (j > 0) hashed = (hashed + hashValues[i][j - 1]) % mod;
                hashValues[i][j] = hashed;
            }
        }
    }

    ll substring_hash(int l, int r) const {
        vector<ll> hash_result(hashPrimes.size(), 0);
        for (int i = 0; i < (int)hashPrimes.size(); i++) {
            ll mod = hashPrimes[i];
            ll val1 = hashValues[i][r];
            ll val2 = l > 0 ? hashValues[i][l - 1] : 0;
            ll sub_hash = (val1 - val2 + mod) % mod;
            hash_result[i] = sub_hash * inversePowersOfBase[i][l] % mod;
        }
        return 1000000021LL * hash_result[0] + hash_result[1];
    }
};

vector<int> kmp_search(const string& text, const string& pattern) {
    if (pattern.empty() || text.empty()) return {};
    vector<int> lps(pattern.size(), 0);
    int len = 0;
    for (int i = 1; i < (int)pattern.size(); i++) {
        while (len > 0 && pattern[i] != pattern[len]) len = lps[len - 1];
        if (pattern[i] == pattern[len]) lps[i] = ++len;
    }

    vector<int> result;
    int i = 0, j = 0;
    while (i < (int)text.size()) {
        if (text[i] == pattern[j]) {
            i++;
            j++;
            if (j == (int)pattern.size()) {
                result.push_back(i - j);
                j = lps[j - 1];
            }
        } else if (j > 0) {
            j = lps[j - 1];
        } else {
            i++;
        }
    }
    return result;
}

struct AhoCorasick {
    vector<unordered_map<char, int>> trie;
    vector<int> fail;
    vector<set<int>> output;

    AhoCorasick() {
        trie.emplace_back();
        fail.push_back(0);
        output.emplace_back();
    }

    void add(const string& word, int idx) {
        int node = 0;
        for (char c : word) {
            if (!trie[node].count(c)) {
                trie[node][c] = (int)trie.size();
                trie.emplace_back();
                fail.push_back(0);
                output.emplace_back();
            }
            node = trie[node][c];
        }
        output[node].insert(idx);
    }

    void build() {
        queue<int> q;
        for (auto [c, nxt] : trie[0]) {
            q.push(nxt);
            fail[nxt] = 0;
        }

        while (!q.empty()) {
            int node = q.front();
            q.pop();
            for (auto [c, nxt] : trie[node]) {
                int f = fail[node];
                while (f && !trie[f].count(c)) f = fail[f];
                fail[nxt] = trie[f].count(c) ? trie[f][c] : 0;
                output[nxt].insert(output[fail[nxt]].begin(), output[fail[nxt]].end());
                q.push(nxt);
            }
        }
    }

    vector<pair<int, int>> search(const string& text) const {
        int node = 0;
        vector<pair<int, int>> results;
        for (int i = 0; i < (int)text.size(); i++) {
            char c = text[i];
            while (node && !trie[node].count(c)) node = fail[node];
            if (trie[node].count(c)) node = trie[node].at(c);
            else node = 0;
            for (int pat_idx : output[node]) results.push_back({i, pat_idx});
        }
        return results;
    }
};

tuple<int, int, int> manacher(const string& s) {
    string t = "^";
    for (char c : s) {
        t += "#";
        t += c;
    }
    t += "#$";

    int n = (int)t.size();
    vector<int> p(n, 0);
    int c = 0, r = 0;
    for (int i = 1; i < n - 1; i++) {
        int mirror = 2 * c - i;
        if (i < r) p[i] = min(r - i, p[mirror]);
        while (t[i + 1 + p[i]] == t[i - 1 - p[i]]) p[i]++;
        if (i + p[i] > r) {
            c = i;
            r = i + p[i];
        }
    }

    int max_len = 0, center = 0;
    for (int i = 0; i < n; i++) {
        if (p[i] > max_len) {
            max_len = p[i];
            center = i;
        }
    }
    int start = (center - max_len) / 2;
    int end = start + max_len - 1;
    return {max_len, start, end};
}
