#include <bits/stdc++.h>
using namespace std;

using ll = long long;

vector<int> euler_sieve(int n) {
    vector<bool> is_prime(n + 1, true);
    vector<int> primes;
    if (n >= 0) is_prime[0] = false;
    if (n >= 1) is_prime[1] = false;
    for (int i = 2; i <= n; i++) {
        if (is_prime[i]) primes.push_back(i);
        for (int p : primes) {
            if (1LL * i * p > n) break;
            is_prime[i * p] = false;
            if (i % p == 0) break;
        }
    }
    return primes;
}

pair<vector<int>, vector<int>> linear_sieve_lp(int N) {
    vector<int> lp(N + 1, 0), pr;
    for (int i = 2; i <= N; i++) {
        if (lp[i] == 0) {
            lp[i] = i;
            pr.push_back(i);
        }
        for (int p : pr) {
            if (p > lp[i] || 1LL * i * p > N) break;
            lp[i * p] = p;
        }
    }
    return {lp, pr};
}

vector<pair<int, int>> get_prime_factors(int n, const vector<int>& lp) {
    vector<pair<int, int>> factors;
    while (n > 1) {
        int p = lp[n], cnt = 0;
        while (n % p == 0) {
            n /= p;
            cnt++;
        }
        factors.push_back({p, cnt});
    }
    return factors;
}

tuple<ll, ll, ll> egcd_iterative(ll a, ll b) {
    ll A = a, B = b;
    ll x0 = 1, y0 = 0, x1 = 0, y1 = 1;
    while (B != 0) {
        ll q = A / B;
        tie(A, B) = pair<ll, ll>{B, A - q * B};
        tie(x0, x1) = pair<ll, ll>{x1, x0 - q * x1};
        tie(y0, y1) = pair<ll, ll>{y1, y0 - q * y1};
    }
    if (A < 0) {
        A = -A;
        x0 = -x0;
        y0 = -y0;
    }
    return {A, x0, y0};
}

ll phi(ll n, const vector<int>& primes) {
    ll result = n, temp = n;
    for (ll p : primes) {
        if (p * p > temp) break;
        if (temp % p == 0) {
            while (temp % p == 0) temp /= p;
            result -= result / p;
        }
    }
    if (temp > 1) result -= result / temp;
    return result;
}

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

ll mod_inv(ll a, ll mod) {
    auto [g, x, y] = egcd_iterative(a, mod);
    assert(g == 1);
    x %= mod;
    if (x < 0) x += mod;
    return x;
}

pair<ll, ll> crt(const vector<ll>& remainders, const vector<ll>& moduli) {
    assert(remainders.size() == moduli.size());
    ll M = 1;
    for (ll m : moduli) M *= m;
    ll x = 0;
    for (int i = 0; i < (int)remainders.size(); i++) {
        ll ai = remainders[i], mi = moduli[i];
        ll Mi = M / mi;
        ll yi = mod_inv(Mi % mi, mi);
        x = (x + (__int128)ai * Mi % M * yi) % M;
    }
    return {x, M};
}

pair<ll, ll> crt_general(ll a, ll n, ll b, ll m) {
    auto [g, x, y] = egcd_iterative(n, m);
    if ((b - a) % g != 0) return {0, -1};
    ll lcm = n / g * m;
    ll mul = (__int128)((b - a) / g) * x % (m / g);
    ll res = (a + (__int128)n * mul) % lcm;
    if (res < 0) res += lcm;
    return {res, lcm};
}

vector<int> mobius_sieve(int n) {
    vector<int> mu(n + 1, 1), primes;
    vector<bool> is_prime(n + 1, true);
    if (n >= 0) mu[0] = 0;
    for (int i = 2; i <= n; i++) {
        if (is_prime[i]) {
            primes.push_back(i);
            mu[i] = -1;
        }
        for (int p : primes) {
            if (1LL * i * p > n) break;
            is_prime[i * p] = false;
            if (i % p == 0) {
                mu[i * p] = 0;
                break;
            }
            mu[i * p] = -mu[i];
        }
    }
    return mu;
}

struct CoprimeCounter {
    map<ll, pair<vector<ll>, vector<ll>>> cache;

    vector<ll> factor(ll k) {
        vector<ll> pf;
        ll x = k;
        for (ll d = 2; d * d <= x; d++) {
            if (x % d == 0) {
                pf.push_back(d);
                while (x % d == 0) x /= d;
            }
        }
        if (x > 1) pf.push_back(x);
        return pf;
    }

    vector<ll> gen_divisors(const vector<ll>& pf) {
        vector<ll> divs = {1};
        for (ll p : pf) {
            int sz = (int)divs.size();
            for (int i = 0; i < sz; i++) divs.push_back(divs[i] * p);
        }
        return divs;
    }

    int mobius_of_divisor(ll d, const vector<ll>& pf) {
        int cnt = 0;
        for (ll p : pf) {
            if (d % p == 0) cnt++;
        }
        return (cnt & 1) ? -1 : 1;
    }

    pair<vector<ll>, vector<ll>> data(ll k) {
        if (!cache.count(k)) {
            vector<ll> pf = factor(k);
            cache[k] = {pf, gen_divisors(pf)};
        }
        return cache[k];
    }

    ll count(ll n, ll k) {
        if (n <= 0) return 0;
        if (k < 0) k = -k;
        if (k == 0) return n >= 1 ? 1 : 0;
        if (k == 1) return n;
        auto [pf, divs] = data(k);
        ll ans = 0;
        for (ll d : divs) ans += mobius_of_divisor(d, pf) * (n / d);
        return ans;
    }
};

bool is_prime(ll n) {
    if (n < 2) return false;
    for (ll p : {2LL, 3LL, 5LL, 7LL, 11LL, 13LL, 17LL, 19LL, 23LL, 29LL}) {
        if (n % p == 0) return n == p;
    }
    ll d = n - 1, s = 0;
    while ((d & 1) == 0) {
        d >>= 1;
        s++;
    }
    auto check = [&](ll a) {
        ll x = mod_pow(a, d, n);
        if (x == 1 || x == n - 1) return true;
        for (int r = 0; r < s - 1; r++) {
            x = (__int128)x * x % n;
            if (x == n - 1) return true;
        }
        return false;
    };
    for (ll a : {2LL, 3LL, 5LL, 7LL, 11LL, 13LL}) {
        if (a % n == 0) continue;
        if (!check(a)) return false;
    }
    return true;
}
