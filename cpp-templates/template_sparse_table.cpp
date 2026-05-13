#include <bits/stdc++.h>
using namespace std;

template <class T, class Op>
struct SparseTable {
    int N, K;
    Op func;
    vector<int> lg;
    vector<vector<T>> st;

    SparseTable(const vector<T>& arr, Op func_) : func(func_) {
        N = (int)arr.size();
        K = N ? 32 - __builtin_clz(N) : 1;
        lg.assign(N + 1, 0);
        for (int i = 2; i <= N; i++) lg[i] = lg[i / 2] + 1;
        st.assign(N, vector<T>(K));
        for (int i = 0; i < N; i++) st[i][0] = arr[i];
        for (int j = 1; j < K; j++) {
            for (int i = 0; i + (1 << j) <= N; i++) {
                st[i][j] = func(st[i][j - 1], st[i + (1 << (j - 1))][j - 1]);
            }
        }
    }

    T query(int L, int R) const {
        int j = lg[R - L + 1];
        return func(st[L][j], st[R - (1 << j) + 1][j]);
    }
};

template <class T, class Op>
struct SparseTable2D {
    int N, M, K1, K2;
    Op func;
    vector<int> logN, logM;
    vector<vector<vector<vector<T>>>> st;

    SparseTable2D(const vector<vector<T>>& matrix, Op func_) : func(func_) {
        N = (int)matrix.size();
        M = N ? (int)matrix[0].size() : 0;
        K1 = N ? 32 - __builtin_clz(N) : 1;
        K2 = M ? 32 - __builtin_clz(M) : 1;
        logN.assign(N + 1, 0);
        logM.assign(M + 1, 0);
        for (int i = 2; i <= N; i++) logN[i] = logN[i / 2] + 1;
        for (int j = 2; j <= M; j++) logM[j] = logM[j / 2] + 1;

        st.assign(N, vector<vector<vector<T>>>(M, vector<vector<T>>(K1, vector<T>(K2))));
        build(matrix);
    }

    void build(const vector<vector<T>>& matrix) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < M; j++) st[i][j][0][0] = matrix[i][j];
        }
        for (int k2 = 1; k2 < K2; k2++) {
            int step = 1 << (k2 - 1);
            for (int i = 0; i < N; i++) {
                for (int j = 0; j + (1 << k2) <= M; j++) {
                    st[i][j][0][k2] = func(st[i][j][0][k2 - 1], st[i][j + step][0][k2 - 1]);
                }
            }
        }
        for (int k1 = 1; k1 < K1; k1++) {
            int step = 1 << (k1 - 1);
            for (int i = 0; i + (1 << k1) <= N; i++) {
                for (int j = 0; j < M; j++) {
                    for (int k2 = 0; k2 < K2; k2++) {
                        st[i][j][k1][k2] = func(st[i][j][k1 - 1][k2], st[i + step][j][k1 - 1][k2]);
                    }
                }
            }
        }
    }

    T query(int x1, int y1, int x2, int y2) const {
        int k1 = logN[x2 - x1 + 1];
        int k2 = logM[y2 - y1 + 1];
        int xs = 1 << k1, ys = 1 << k2;
        return func(
            func(st[x1][y1][k1][k2], st[x2 - xs + 1][y1][k1][k2]),
            func(st[x1][y2 - ys + 1][k1][k2], st[x2 - xs + 1][y2 - ys + 1][k1][k2])
        );
    }
};
