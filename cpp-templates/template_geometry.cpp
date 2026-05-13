#include <bits/stdc++.h>
using namespace std;

const double GEOM_EPS = 1e-9;

struct Point {
    double x, y;

    bool operator<(const Point& other) const {
        if (fabs(x - other.x) > GEOM_EPS) return x < other.x;
        return y < other.y - GEOM_EPS;
    }

    bool operator==(const Point& other) const {
        return fabs(x - other.x) <= GEOM_EPS && fabs(y - other.y) <= GEOM_EPS;
    }
};

double cross(Point o, Point a, Point b) {
    return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
}

double dist(Point a, Point b) {
    return hypot(a.x - b.x, a.y - b.y);
}

double polygon_area(const vector<Point>& points) {
    int n = (int)points.size();
    double area = 0;
    for (int i = 0; i < n; i++) {
        Point a = points[i], b = points[(i + 1) % n];
        area += a.x * b.y - b.x * a.y;
    }
    return fabs(area) / 2.0;
}

bool point_in_polygon(Point p, const vector<Point>& poly) {
    bool inside = false;
    int n = (int)poly.size();
    for (int i = 0; i < n; i++) {
        Point a = poly[i], b = poly[(i + 1) % n];
        bool crosses = ((a.y > p.y) != (b.y > p.y));
        if (crosses && p.x < (b.x - a.x) * (p.y - a.y) / (b.y - a.y + 1e-12) + a.x) {
            inside = !inside;
        }
    }
    return inside;
}

int orientation(Point a, Point b, Point c) {
    double val = (b.y - a.y) * (c.x - b.x) - (b.x - a.x) * (c.y - b.y);
    if (fabs(val) < GEOM_EPS) return 0;
    return val > 0 ? 1 : 2;
}

bool on_segment(Point a, Point b, Point c) {
    return min(a.x, c.x) - GEOM_EPS <= b.x && b.x <= max(a.x, c.x) + GEOM_EPS &&
           min(a.y, c.y) - GEOM_EPS <= b.y && b.y <= max(a.y, c.y) + GEOM_EPS;
}

bool segments_intersect(Point p1, Point q1, Point p2, Point q2) {
    int o1 = orientation(p1, q1, p2);
    int o2 = orientation(p1, q1, q2);
    int o3 = orientation(p2, q2, p1);
    int o4 = orientation(p2, q2, q1);

    if (o1 != o2 && o3 != o4) return true;
    if (o1 == 0 && on_segment(p1, p2, q1)) return true;
    if (o2 == 0 && on_segment(p1, q2, q1)) return true;
    if (o3 == 0 && on_segment(p2, p1, q2)) return true;
    if (o4 == 0 && on_segment(p2, q1, q2)) return true;
    return false;
}

double closest_pair_rec(vector<Point>& pts, int l, int r) {
    if (r - l <= 3) {
        double ans = numeric_limits<double>::infinity();
        for (int i = l; i < r; i++) {
            for (int j = i + 1; j < r; j++) ans = min(ans, dist(pts[i], pts[j]));
        }
        sort(pts.begin() + l, pts.begin() + r, [](Point a, Point b) { return a.y < b.y; });
        return ans;
    }
    int m = (l + r) / 2;
    double midx = pts[m].x;
    double d = min(closest_pair_rec(pts, l, m), closest_pair_rec(pts, m, r));
    inplace_merge(pts.begin() + l, pts.begin() + m, pts.begin() + r,
                  [](Point a, Point b) { return a.y < b.y; });

    vector<Point> strip;
    for (int i = l; i < r; i++) {
        if (fabs(pts[i].x - midx) < d) strip.push_back(pts[i]);
    }
    for (int i = 0; i < (int)strip.size(); i++) {
        for (int j = i + 1; j < (int)strip.size() && j <= i + 7; j++) {
            d = min(d, dist(strip[i], strip[j]));
        }
    }
    return d;
}

double closest_pair(vector<Point> points) {
    sort(points.begin(), points.end());
    return closest_pair_rec(points, 0, (int)points.size());
}

vector<Point> convex_hull(vector<Point> points) {
    sort(points.begin(), points.end());
    points.erase(unique(points.begin(), points.end()), points.end());
    if (points.size() <= 1) return points;

    vector<Point> lower, upper;
    for (Point p : points) {
        while (lower.size() >= 2 && cross(lower[lower.size() - 2], lower.back(), p) <= GEOM_EPS) {
            lower.pop_back();
        }
        lower.push_back(p);
    }
    for (int i = (int)points.size() - 1; i >= 0; i--) {
        Point p = points[i];
        while (upper.size() >= 2 && cross(upper[upper.size() - 2], upper.back(), p) <= GEOM_EPS) {
            upper.pop_back();
        }
        upper.push_back(p);
    }
    lower.pop_back();
    upper.pop_back();
    lower.insert(lower.end(), upper.begin(), upper.end());
    return lower;
}
