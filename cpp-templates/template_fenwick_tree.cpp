#include <bits/stdc++.h>
using namespace std;

using ll = long long;

struct FenwickTree {
    int n;
    vector<ll> bit;

    FenwickTree(int n = 0) { init(n); }

    void init(int n_) {
        n = n_;
        bit.assign(n + 1, 0);
    }

    void add(int idx, ll delta) {
        for (; idx <= n; idx += idx & -idx) bit[idx] += delta;
    }

    ll prefix_sum(int idx) const {
        ll res = 0;
        for (; idx > 0; idx -= idx & -idx) res += bit[idx];
        return res;
    }

    ll range_sum(int l, int r) const {
        return prefix_sum(r) - prefix_sum(l - 1);
    }

    int find_kth(ll k) const {
        int idx = 0;
        int bit_mask = 1;
        while ((bit_mask << 1) <= n) bit_mask <<= 1;
        for (; bit_mask; bit_mask >>= 1) {
            int nxt = idx + bit_mask;
            if (nxt <= n && bit[nxt] < k) {
                idx = nxt;
                k -= bit[nxt];
            }
        }
        return idx + 1;
    }
};

struct DoubleFenwick {
    int n;
    FenwickTree B1, B2;

    DoubleFenwick(int n = 0) { init(n); }

    void init(int n_) {
        n = n_;
        B1.init(n);
        B2.init(n);
    }

    void add_safe(FenwickTree& ft, int idx, ll delta) {
        if (idx <= n) ft.add(idx, delta);
    }

    void range_add(int l, int r, ll delta) {
        add_safe(B1, l, delta);
        add_safe(B1, r + 1, -delta);
        add_safe(B2, l, delta * (l - 1));
        add_safe(B2, r + 1, -delta * r);
    }

    ll prefix_sum(int idx) const {
        return B1.prefix_sum(idx) * idx - B2.prefix_sum(idx);
    }

    ll range_sum(int l, int r) const {
        return prefix_sum(r) - prefix_sum(l - 1);
    }
};

struct FenwickTree2D {
    int n, m;
    vector<vector<ll>> bit;

    FenwickTree2D(int n = 0, int m = 0) { init(n, m); }

    void init(int n_, int m_) {
        n = n_;
        m = m_;
        bit.assign(n + 1, vector<ll>(m + 1, 0));
    }

    void add(int x, int y, ll delta) {
        for (int i = x; i <= n; i += i & -i) {
            for (int j = y; j <= m; j += j & -j) {
                bit[i][j] += delta;
            }
        }
    }

    ll sum(int x, int y) const {
        ll res = 0;
        for (int i = x; i > 0; i -= i & -i) {
            for (int j = y; j > 0; j -= j & -j) {
                res += bit[i][j];
            }
        }
        return res;
    }

    ll range_sum(int x1, int y1, int x2, int y2) const {
        return sum(x2, y2) - sum(x1 - 1, y2) - sum(x2, y1 - 1) + sum(x1 - 1, y1 - 1);
    }
};
