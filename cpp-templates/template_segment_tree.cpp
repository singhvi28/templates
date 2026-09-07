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

template <class S,
          S (*op)(S, S),
          S (*e)(),
          class F,
          S (*mapping)(F, S),
          F (*composition)(F, F),
          F (*id)()>
struct LazySegTree {
    int _n, size, log;
    std::vector<S> d;
    std::vector<F> lz;

    LazySegTree() : LazySegTree(0) {}
    explicit LazySegTree(int n) : LazySegTree(std::vector<S>(n, e())) {}
    explicit LazySegTree(const std::vector<S>& v) : _n(int(v.size())) {
        size = 1;
        log = 0;
        while (size < _n) {
            size <<= 1;
            log++;
        }
        d.assign(2 * size, e());
        lz.assign(size, id());
        for (int i = 0; i < _n; i++) d[size + i] = v[i];
        for (int i = size - 1; i >= 1; i--) update(i);
    }

    void update(int k) { d[k] = op(d[2 * k], d[2 * k + 1]); }

    void all_apply(int k, F f) {
        d[k] = mapping(f, d[k]);
        if (k < size) lz[k] = composition(f, lz[k]);
    }

    void push(int k) {
        if (lz[k] != id()) {
            all_apply(2 * k, lz[k]);
            all_apply(2 * k + 1, lz[k]);
            lz[k] = id();
        }
    }

    void apply(int p, F f) {
        assert(0 <= p && p < _n);
        p += size;
        for (int i = log; i >= 1; i--) push(p >> i);
        all_apply(p, f);
        for (int i = 1; i <= log; i++) update(p >> i);
    }

    void apply(int l, int r, F f) {
        assert(0 <= l && l <= r && r <= _n);
        if (l == r) return;
        l += size;
        r += size;
        for (int i = log; i >= 1; i--) {
            if (((l >> i) << i) != l) push(l >> i);
            if (((r >> i) << i) != r) push((r - 1) >> i);
        }
        int l2 = l, r2 = r;
        while (l < r) {
            if (l & 1) all_apply(l++, f);
            if (r & 1) all_apply(--r, f);
            l >>= 1;
            r >>= 1;
        }
        l = l2;
        r = r2;
        for (int i = 1; i <= log; i++) {
            if (((l >> i) << i) != l) update(l >> i);
            if (((r >> i) << i) != r) update((r - 1) >> i);
        }
    }

    S prod(int l, int r) {
        assert(0 <= l && l <= r && r <= _n);
        if (l == r) return e();
        l += size;
        r += size;
        for (int i = log; i >= 1; i--) {
            if (((l >> i) << i) != l) push(l >> i);
            if (((r >> i) << i) != r) push((r - 1) >> i);
        }
        S sml = e(), smr = e();
        while (l < r) {
            if (l & 1) sml = op(sml, d[l++]);
            if (r & 1) smr = op(d[--r], smr);
            l >>= 1;
            r >>= 1;
        }
        return op(sml, smr);
    }

    S all_prod() const { return d[1]; }

    S get(int p) {
        assert(0 <= p && p < _n);
        p += size;
        for (int i = log; i >= 1; i--) push(p >> i);
        return d[p];
    }

    void set(int p, S x) {
        assert(0 <= p && p < _n);
        p += size;
        for (int i = log; i >= 1; i--) push(p >> i);
        d[p] = x;
        for (int i = 1; i <= log; i++) update(p >> i);
    }

    template <class G> int find_first(int l, int r, G pred) {
        assert(0 <= l && l <= r && r <= _n);
        if (l == r) return -1;
        l += size;
        r += size;
        for (int i = log; i >= 1; i--) {
            if (((l >> i) << i) != l) push(l >> i);
            if (((r >> i) << i) != r) push((r - 1) >> i);
        }
        std::vector<int> left_nodes, right_nodes;
        while (l < r) {
            if (l & 1) left_nodes.push_back(l++);
            if (r & 1) right_nodes.push_back(--r);
            l >>= 1;
            r >>= 1;
        }
        std::vector<int> canonical = left_nodes;
        for (int i = (int)right_nodes.size() - 1; i >= 0; i--) {
            canonical.push_back(right_nodes[i]);
        }
        for (int node : canonical) {
            if (pred(d[node])) {
                while (node < size) {
                    push(node);
                    if (pred(d[2 * node])) {
                        node = 2 * node;
                    } else {
                        node = 2 * node + 1;
                    }
                }
                return node - size;
            }
        }
        return -1;
    }

    template <class G> int find_last(int l, int r, G pred) {
        assert(0 <= l && l <= r && r <= _n);
        if (l == r) return -1;
        l += size;
        r += size;
        for (int i = log; i >= 1; i--) {
            if (((l >> i) << i) != l) push(l >> i);
            if (((r >> i) << i) != r) push((r - 1) >> i);
        }
        std::vector<int> left_nodes, right_nodes;
        while (l < r) {
            if (l & 1) left_nodes.push_back(l++);
            if (r & 1) right_nodes.push_back(--r);
            l >>= 1;
            r >>= 1;
        }
        std::vector<int> canonical = right_nodes;
        for (int i = (int)left_nodes.size() - 1; i >= 0; i--) {
            canonical.push_back(left_nodes[i]);
        }
        for (int node : canonical) {
            if (pred(d[node])) {
                while (node < size) {
                    push(node);
                    if (pred(d[2 * node + 1])) {
                        node = 2 * node + 1;
                    } else {
                        node = 2 * node;
                    }
                }
                return node - size;
            }
        }
        return -1;
    }

    template <class G> int max_right(int l, G g) {
        assert(0 <= l && l <= _n);
        assert(g(e()));
        if (l == _n) return _n;
        l += size;
        for (int i = log; i >= 1; i--) push(l >> i);
        S sm = e();
        do {
            while (l % 2 == 0) l >>= 1;
            if (!g(op(sm, d[l]))) {
                while (l < size) {
                    push(l);
                    l = (2 * l);
                    if (g(op(sm, d[l]))) {
                        sm = op(sm, d[l]);
                        l++;
                    }
                }
                return l - size;
            }
            sm = op(sm, d[l]);
            l++;
        } while ((l & -l) != l);
        return _n;
    }

    template <class G> int min_left(int r, G g) {
        assert(0 <= r && r <= _n);
        assert(g(e()));
        if (r == 0) return 0;
        r += size;
        for (int i = log; i >= 1; i--) push((r - 1) >> i);
        S sm = e();
        do {
            r--;
            while (r > 1 && (r % 2)) r >>= 1;
            if (!g(op(d[r], sm))) {
                while (r < size) {
                    push(r);
                    r = (2 * r + 1);
                    if (g(op(d[r], sm))) {
                        sm = op(d[r], sm);
                        r--;
                    }
                }
                return r + 1 - size;
            }
            sm = op(d[r], sm);
        } while ((r & -r) != r);
        return 0;
    }
};
