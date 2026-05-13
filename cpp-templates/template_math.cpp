#include <bits/stdc++.h>
using namespace std;

using ll = long long;
const ll MOD = 1000000007LL;

tuple<int, vector<double>> gauss_solve(vector<vector<double>> A, vector<double> b) {
    const double eps = 1e-9;
    int n = (int)A.size();
    int m = A.empty() ? 0 : (int)A[0].size();
    vector<vector<double>> a(n, vector<double>(m + 1));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) a[i][j] = A[i][j];
        a[i][m] = b[i];
    }

    vector<int> pos(m, -1);
    int rank = 0;
    for (int col = 0; col < m; col++) {
        int row = rank, mx = row;
        for (int i = row; i < n; i++) {
            if (fabs(a[i][col]) > fabs(a[mx][col])) mx = i;
        }
        if (mx >= n || fabs(a[mx][col]) < eps) continue;
        swap(a[row], a[mx]);
        pos[col] = row;

        for (int i = 0; i < n; i++) {
            if (i != row && fabs(a[i][col]) > eps) {
                double c = a[i][col] / a[row][col];
                for (int j = col; j <= m; j++) a[i][j] -= a[row][j] * c;
            }
        }
        rank++;
    }

    vector<double> ans(m, 0);
    for (int i = 0; i < m; i++) {
        if (pos[i] != -1) ans[i] = a[pos[i]][m] / a[pos[i]][i];
    }

    for (int i = 0; i < n; i++) {
        double s = 0;
        for (int j = 0; j < m; j++) s += ans[j] * a[i][j];
        if (fabs(s - a[i][m]) > eps) return {-1, {}};
    }
    for (int i = 0; i < m; i++) {
        if (pos[i] == -1) return {2, ans};
    }
    return {1, ans};
}

template <int L, ll MODV = MOD>
struct Mat {
    array<array<ll, L>, L> a{};

    Mat() {
        for (int i = 0; i < L; i++) {
            for (int j = 0; j < L; j++) a[i][j] = 0;
        }
    }

    static Mat identity() {
        Mat m;
        for (int i = 0; i < L; i++) m.a[i][i] = 1;
        return m;
    }

    Mat operator*(const Mat& other) const {
        Mat result;
        for (int i = 0; i < L; i++) {
            for (int k = 0; k < L; k++) {
                if (!a[i][k]) continue;
                for (int j = 0; j < L; j++) {
                    result.a[i][j] = (result.a[i][j] + a[i][k] * other.a[k][j]) % MODV;
                }
            }
        }
        return result;
    }

    Mat transpose() const {
        Mat result;
        for (int i = 0; i < L; i++) {
            for (int j = 0; j < L; j++) result.a[j][i] = a[i][j];
        }
        return result;
    }
};

template <int L, ll MODV = MOD>
Mat<L, MODV> quickmul(Mat<L, MODV> x, long long y) {
    Mat<L, MODV> ans = Mat<L, MODV>::identity();
    while (y) {
        if (y & 1) ans = ans * x;
        x = x * x;
        y >>= 1;
    }
    return ans;
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

struct Combinatorics {
    int N;
    ll mod;
    vector<ll> fact, derangements;

    Combinatorics(int N = 0, ll mod = MOD) { init(N, mod); }

    void init(int N_, ll mod_) {
        N = N_;
        mod = mod_;
        fact.assign(N + 1, 1);
        for (int i = 1; i <= N; i++) fact[i] = fact[i - 1] * i % mod;
        derangements = get_derangements(N);
    }

    ll inversemod(ll a) const {
        return mod_pow(a, mod - 2, mod);
    }

    ll nCr(int n, int r) const {
        if (r > n || n < 0 || r < 0) return 0;
        return fact[n] * inversemod(fact[r]) % mod * inversemod(fact[n - r]) % mod;
    }

    vector<ll> get_derangements(int n) const {
        vector<ll> dp(n + 1, 0);
        if (n >= 0) dp[0] = 1;
        if (n >= 1) dp[1] = 0;
        for (int i = 2; i <= n; i++) dp[i] = (i - 1LL) * (dp[i - 1] + dp[i - 2]) % mod;
        return dp;
    }

    ll derangement(int n) const {
        return derangements[n];
    }
};

struct Basis {
    int B, sz;
    vector<long long> basis;

    Basis(int bit_width = 31) : B(bit_width), sz(0), basis(bit_width, 0) {}

    void clear() {
        fill(basis.begin(), basis.end(), 0);
        sz = 0;
    }

    void insert(long long x) {
        for (int i = B - 1; i >= 0; i--) {
            if ((x >> i) & 1LL) {
                if (basis[i]) x ^= basis[i];
                else {
                    basis[i] = x;
                    sz++;
                    break;
                }
            }
        }
    }

    bool can(long long x) const {
        for (int i = B - 1; i >= 0; i--) x = min(x, x ^ basis[i]);
        return x == 0;
    }

    long long max_xor(long long x = 0) const {
        for (int i = B - 1; i >= 0; i--) x = max(x, x ^ basis[i]);
        return x;
    }

    long long kth(long long k) const {
        if (k < 1 || k > (1LL << sz)) return -1;
        long long x = 0, cnt = 1LL << sz;
        for (int i = B - 1; i >= 0; i--) {
            if (!basis[i]) continue;
            long long limit = cnt >> 1;
            if (k > limit) {
                if (!((x >> i) & 1LL)) x ^= basis[i];
                k -= limit;
            } else {
                if ((x >> i) & 1LL) x ^= basis[i];
            }
            cnt >>= 1;
        }
        return x;
    }

    long long count_lt(long long x) const {
        if (x < 0) return 0;
        long long ans = 0, cnt = 1LL << sz, mask = 0;
        for (int i = B - 1; i >= 0; i--) {
            if (basis[i]) {
                long long half = cnt >> 1;
                if ((x >> i) & 1LL) {
                    ans += half;
                    if (!((mask >> i) & 1LL)) mask ^= basis[i];
                } else if ((mask >> i) & 1LL) {
                    mask ^= basis[i];
                }
                cnt >>= 1;
            } else if (((x >> i) & 1LL) != ((mask >> i) & 1LL)) {
                return ((x >> i) & 1LL) ? ans + cnt : ans;
            }
        }
        return ans;
    }
};
