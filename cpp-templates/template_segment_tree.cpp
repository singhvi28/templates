#include <bits/stdc++.h>
using namespace std;

template <class T, class Op>
struct SegTree {
    int n, size;
    T e;
    Op op;
    vector<T> d;

    SegTree(int n_, Op op_, T e_) : n(n_), e(e_), op(op_) {
        size = 1;
        while (size < n) size <<= 1;
        d.assign(2 * size, e);
    }

    void build(const vector<T>& arr) {
        for (int i = 0; i < (int)arr.size(); i++) d[size + i] = arr[i];
        for (int i = size - 1; i >= 1; i--) d[i] = op(d[2 * i], d[2 * i + 1]);
    }

    void set_val(int p, T x) {
        p += size;
        d[p] = x;
        while (p > 1) {
            p >>= 1;
            d[p] = op(d[2 * p], d[2 * p + 1]);
        }
    }

    T get(int p) const {
        return d[p + size];
    }

    T prod(int l, int r) const {
        T sml = e, smr = e;
        l += size;
        r += size;
        while (l < r) {
            if (l & 1) sml = op(sml, d[l++]);
            if (r & 1) smr = op(d[--r], smr);
            l >>= 1;
            r >>= 1;
        }
        return op(sml, smr);
    }
};

struct MergeSortTree {
    int n;
    vector<vector<long long>> v, pref;

    MergeSortTree(const vector<long long>& a = {}) { build(a); }

    void build(const vector<long long>& a) {
        n = (int)a.size();
        v.assign(max(1, 4 * n), {});
        pref.assign(max(1, 4 * n), {});
        if (n > 0) build_rec(1, 0, n - 1, a);
    }

    void build_rec(int idx, int l, int r, const vector<long long>& a) {
        if (l == r) {
            v[idx] = {a[l]};
            pref[idx] = {a[l]};
            return;
        }
        int m = (l + r) / 2;
        build_rec(idx * 2, l, m, a);
        build_rec(idx * 2 + 1, m + 1, r, a);
        merge(v[idx * 2].begin(), v[idx * 2].end(), v[idx * 2 + 1].begin(), v[idx * 2 + 1].end(),
              back_inserter(v[idx]));
        pref[idx].resize(v[idx].size());
        partial_sum(v[idx].begin(), v[idx].end(), pref[idx].begin());
    }

    pair<int, long long> le_rec(int idx, int l, int r, int L, int R, long long x) const {
        if (R < l || r < L) return {0, 0};
        if (L <= l && r <= R) {
            int pos = upper_bound(v[idx].begin(), v[idx].end(), x) - v[idx].begin();
            long long sum = pos > 0 ? pref[idx][pos - 1] : 0;
            return {pos, sum};
        }
        int m = (l + r) / 2;
        auto a = le_rec(idx * 2, l, m, L, R, x);
        auto b = le_rec(idx * 2 + 1, m + 1, r, L, R, x);
        return {a.first + b.first, a.second + b.second};
    }

    pair<int, long long> le(int L, int R, long long x) const {
        if (n == 0) return {0, 0};
        return le_rec(1, 0, n - 1, L, R, x);
    }

    long long sum_rec(int idx, int l, int r, int L, int R) const {
        if (R < l || r < L) return 0;
        if (L <= l && r <= R) return pref[idx].empty() ? 0 : pref[idx].back();
        int m = (l + r) / 2;
        return sum_rec(idx * 2, l, m, L, R) + sum_rec(idx * 2 + 1, m + 1, r, L, R);
    }

    long long range_sum(int L, int R) const {
        if (n == 0) return 0;
        return sum_rec(1, 0, n - 1, L, R);
    }
};
