#include <bits/stdc++.h>
using namespace std;

class ConvexHullTrick {
private:
    // Stores pairs of (m, c) representing the line y = mx + c
    deque<pair<long long, long long>> lines;

    bool is_redundant(const pair<long long, long long>& l1,
                      const pair<long long, long long>& l2,
                      const pair<long long, long long>& l3) {
        /*
            Helper method to check if line l2 is rendered useless by line l3.
            Avoids division to prevent floating point precision issues.

            Mathematically equivalent to:
            intersection(l1, l3) <= intersection(l1, l2)
        */
        long long m1 = l1.first, c1 = l1.second;
        long long m2 = l2.first, c2 = l2.second;
        long long m3 = l3.first, c3 = l3.second;

        // Use __int128 to reduce overflow risk during multiplication.
        __int128 left  = (__int128)(c3 - c1) * (m1 - m2);
        __int128 right = (__int128)(c2 - c1) * (m1 - m3);
        return left <= right;
    }

public:
    ConvexHullTrick() = default;

    void add(long long m, long long c) {
        /*
            Add the line y = mx + c to the hull.
            RULE: Slopes (m) must be added in strictly decreasing order.
        */
        pair<long long, long long> l3 = {m, c};

        // Remove lines from the back that are no longer part of the optimal "envelope"
        while (lines.size() >= 2 && is_redundant(lines[lines.size() - 2], lines.back(), l3)) {
            lines.pop_back();
        }

        lines.push_back(l3);
    }

    long long query(long long x) {
        /*
            Get the minimum y value across all lines at a specific x.
            RULE: Queries (x) must be monotonically increasing.
        */
        if (lines.empty()) {
            return numeric_limits<long long>::max(); // Or whatever your default maximum should be
        }

        // Since x is increasing, if line 0 is worse than line 1 at this x,
        // line 0 will NEVER be optimal for any future x. We can safely throw it away.
        while (lines.size() >= 2) {
            long long m1 = lines[0].first, c1 = lines[0].second;
            long long m2 = lines[1].first, c2 = lines[1].second;

            __int128 y1 = (__int128)m1 * x + c1;
            __int128 y2 = (__int128)m2 * x + c2;

            if (y1 >= y2) {
                lines.pop_front();
            } else {
                break;
            }
        }

        // The line at the front of the deque is guaranteed to be the best for this x
        long long best_m = lines[0].first;
        long long best_c = lines[0].second;
        __int128 ans = (__int128)best_m * x + best_c;
        return (long long)ans;
    }
};

// // 1. Initialize the Black Box
// ConvexHullTrick cht;
//
// // Let's say we have these lines (slopes are decreasing: 5, 3, 1)
// // y = 5x + 2
// // y = 3x + 10
// // y = 1x + 25
//
// // 2. Add lines (respecting the rule: slopes must be decreasing)
// cht.add(5, 2);
// cht.add(3, 10);
// cht.add(1, 25);
//
// // 3. Query minimums (respecting the rule: x must be increasing)
// cout << cht.query(1) << '\n';  // Output: 7   (from 5*1 + 2)
// cout << cht.query(3) << '\n';  // Output: 17  (from 5*3 + 2, wait! 3*3+10 is 19. So 5x+2 is still best)
// cout << cht.query(5) << '\n';  // Output: 25  (from 3*5 + 10)
// cout << cht.query(8) << '\n';  // Output: 33  (from 1*8 + 25)
