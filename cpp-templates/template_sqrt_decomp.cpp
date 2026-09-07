#include <bits/stdc++.h>
using namespace std;

// Customization Checklist:
// build(): Set up your aggregate variables (sums, maxes, frequency counters, etc.)
// apply_lazy(val): Define how a block's aggregate and lazy tag change on a full block update.
// push(): Define how a pending lazy tag is distributed to elements.
// update_element(idx, val): Modify a single element.
// query_full() / query_element(idx): Return answers.

struct Block {
    vector<long long> vals;
    long long lazy;
    long long aggregate;

    Block() : lazy(0), aggregate(0) {}
    Block(const vector<long long>& v) : vals(v), lazy(0), aggregate(0) {
        build();
    }

    void build() {
        // Recompute aggregate data from scratch
        // e.g. aggregate = 0; for(auto x : vals) aggregate += x;
    }

    void push() {
        if (!lazy) return;
        // e.g. for(int i=0; i<vals.size(); i++) vals[i] += lazy;
        lazy = 0;
    }

    void apply_lazy(long long val) {
        // e.g. lazy += val; aggregate += val * vals.size();
    }

    void update_element(int idx, long long val) {
        // e.g. vals[idx] += val;
    }

    long long query_full() const {
        return aggregate;
    }

    long long query_element(int idx) const {
        return vals[idx];
    }
};

struct SqrtDecomp {
    int B;
    vector<Block> blocks;

    SqrtDecomp(const vector<long long>& arr = {}) {
        int n = (int)arr.size();
        if (n == 0) return;
        B = max(1, (int)sqrt(n) + 1);
        for (int i = 0; i < n; i += B) {
            int end_idx = min(n, i + B);
            vector<long long> sub(arr.begin() + i, arr.begin() + end_idx);
            blocks.emplace_back(sub);
        }
    }

    void range_update(int l, int r, long long val) {
        for (int bi = 0; bi < (int)blocks.size(); bi++) {
            int start = bi * B;
            int end = start + (int)blocks[bi].vals.size();

            if (r < start || l >= end) continue;

            if (l <= start && end - 1 <= r) {
                blocks[bi].apply_lazy(val);
                continue;
            }

            blocks[bi].push();
            int L = max(l, start) - start;
            int R = min(r + 1, end) - start;
            for (int i = L; i < R; i++) {
                blocks[bi].update_element(i, val);
            }
            blocks[bi].build();
        }
    }

    long long range_query(int l, int r) {
        long long res = 0; // Identity element
        for (int bi = 0; bi < (int)blocks.size(); bi++) {
            int start = bi * B;
            int end = start + (int)blocks[bi].vals.size();

            if (r < start || l >= end) continue;

            if (l <= start && end - 1 <= r) {
                // res += blocks[bi].query_full();
                continue;
            }

            blocks[bi].push();
            int L = max(l, start) - start;
            int R = min(r + 1, end) - start;
            for (int i = L; i < R; i++) {
                // res += blocks[bi].query_element(i);
            }
        }
        return res;
    }
};
