#include <bits/stdc++.h>
using namespace std;

template <class Check>
long long find_smallest_true(long long lo, long long hi, Check check) {
    long long ans = -1;
    while (lo <= hi) {
        long long mid = lo + (hi - lo) / 2;
        if (check(mid)) {
            ans = mid;
            hi = mid - 1;
        } else {
            lo = mid + 1;
        }
    }
    return ans;
}

template <class Check>
long long find_largest_true(long long lo, long long hi, Check check) {
    long long ans = -1;
    while (lo <= hi) {
        long long mid = lo + (hi - lo) / 2;
        if (check(mid)) {
            ans = mid;
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    return ans;
}

template <class F>
long long ternary_search_int(long long lo, long long hi, F f) {
    while (hi - lo > 3) {
        long long m1 = lo + (hi - lo) / 3;
        long long m2 = hi - (hi - lo) / 3;
        if (f(m1) < f(m2)) hi = m2 - 1;
        else lo = m1 + 1;
    }
    long long best = lo;
    for (long long x = lo; x <= hi; x++) {
        if (f(x) < f(best)) best = x;
    }
    return best;
}

template <class F>
double ternary_search_double(double left, double right, F f, double eps = 1e-7) {
    while (right - left > eps) {
        double m1 = left + (right - left) / 3.0;
        double m2 = right - (right - left) / 3.0;
        if (f(m1) < f(m2)) right = m2;
        else left = m1;
    }
    return (left + right) / 2.0;
}
