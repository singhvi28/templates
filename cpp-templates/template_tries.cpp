#include <bits/stdc++.h>
using namespace std;

struct BinaryTrie {
    struct Node {
        int child[2];
        bool is_end;
        Node() : child{-1, -1}, is_end(false) {}
    };

    vector<Node> trie;
    int bit_width;

    BinaryTrie(int bit_width_ = 32) : bit_width(bit_width_) {
        trie.emplace_back();
    }

    void insert(int number) {
        int node = 0;
        for (int i = bit_width - 1; i >= 0; i--) {
            int bit = (number >> i) & 1;
            if (trie[node].child[bit] == -1) {
                trie[node].child[bit] = (int)trie.size();
                trie.emplace_back();
            }
            node = trie[node].child[bit];
        }
        trie[node].is_end = true;
    }

    bool search(int number) const {
        int node = 0;
        for (int i = bit_width - 1; i >= 0; i--) {
            int bit = (number >> i) & 1;
            if (trie[node].child[bit] == -1) return false;
            node = trie[node].child[bit];
        }
        return trie[node].is_end;
    }

    optional<int> find_max_xor(int number) const {
        if (trie[0].child[0] == -1 && trie[0].child[1] == -1) return nullopt;
        int node = 0, max_xor = 0;
        for (int i = bit_width - 1; i >= 0; i--) {
            int bit = (number >> i) & 1;
            int want = bit ^ 1;
            if (trie[node].child[want] != -1) {
                max_xor |= (1 << i);
                node = trie[node].child[want];
            } else {
                node = trie[node].child[bit];
            }
        }
        return max_xor;
    }
};
