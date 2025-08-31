from typing import List, Tuple
import math

Point = Tuple[float, float]

def polygon_area(points: List[Point]) -> float:
    """
    Compute the area of a polygon using the Shoelace formula.
    Args: points -> list of vertices in order (x, y).
    Returns: area (float).
    Authored By: Akshit Singhvi
    """
    n = len(points)
    area = 0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i+1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def point_in_polygon(point: Point, polygon: List[Point]) -> bool:
    """
    Check if a point lies inside a polygon using Ray Casting.
    Args: point -> (x, y), polygon -> list of vertices (x, y).
    Returns: True if inside, False otherwise.
    Authored By: Akshit Singhvi
    """
    x, y = point
    n = len(polygon)
    inside = False
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i+1) % n]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1):
            inside = not inside
    return inside


def segments_intersect(p1: Point, q1: Point, p2: Point, q2: Point) -> bool:
    """
    Check if two line segments intersect.
    Args: p1, q1, p2, q2 -> segment endpoints (x, y).
    Returns: True if they intersect, False otherwise.
    Authored By: Akshit Singhvi
    """
    def orientation(a, b, c):
        val = (b[1]-a[1])*(c[0]-b[0]) - (b[0]-a[0])*(c[1]-b[1])
        if abs(val) < 1e-9: return 0
        return 1 if val > 0 else 2

    def on_segment(a, b, c):
        return min(a[0],c[0]) <= b[0] <= max(a[0],c[0]) and min(a[1],c[1]) <= b[1] <= max(a[1],c[1])

    o1 = orientation(p1,q1,p2)
    o2 = orientation(p1,q1,q2)
    o3 = orientation(p2,q2,p1)
    o4 = orientation(p2,q2,q1)

    if o1 != o2 and o3 != o4: return True
    if o1 == 0 and on_segment(p1,p2,q1): return True
    if o2 == 0 and on_segment(p1,q2,q1): return True
    if o3 == 0 and on_segment(p2,p1,q2): return True
    if o4 == 0 and on_segment(p2,q1,q2): return True
    return False


def closest_pair(points: List[Point]) -> float:
    """
    Find the minimum Euclidean distance between any two points.
    Args: points -> list of (x, y).
    Returns: minimum distance (float).
    Authored By: Akshit Singhvi
    """
    def dist(p1, p2):
        return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

    def closest_pair_rec(pts):
        if len(pts) <= 3:
            return min((dist(pts[i], pts[j]) for i in range(len(pts)) for j in range(i+1,len(pts))), default=float('inf'))
        mid = len(pts)//2
        midx = pts[mid][0]
        d = min(closest_pair_rec(pts[:mid]), closest_pair_rec(pts[mid:]))
        strip = [p for p in pts if abs(p[0]-midx) < d]
        strip.sort(key=lambda p: p[1])
        for i in range(len(strip)):
            for j in range(i+1, min(i+7, len(strip))):
                d = min(d, dist(strip[i], strip[j]))
        return d

    pts_sorted = sorted(points)
    return closest_pair_rec(pts_sorted)


def convex_hull(points: List[Point]) -> List[Point]:
    """
    Compute the convex hull of a set of points using Andrew's Monotone Chain.
    Args: points -> list of (x, y).
    Returns: list of hull vertices in CCW order.
    Authored By: Akshit Singhvi
    """
    points = sorted(set(points))
    if len(points) <= 1:
        return points

    def cross(o,a,b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

    lower, upper = [], []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]
