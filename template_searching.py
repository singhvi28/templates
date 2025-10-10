# BINARY SEARCH FUNCTIONS

# returns -1 if no value satisfies

def find_smallest_true(lo, hi, check):
    left, right = lo, hi
    ans = -1
    while left <= right:
        mid = (left + right) // 2
        if check(mid):
            ans = mid
            right = mid - 1
        else:
            left = mid + 1
    return ans

def find_largest_true(lo, hi, check):
    left, right = lo, hi
    ans = -1
    while left <= right:
        mid = (left + right) // 2
        if check(mid):
            ans = mid
            left = mid + 1
        else:
            right = mid - 1
    return ans


# TERNARY SEARCH FOR MINIMUM (INTEGER VERSION)

def ternary_search(lo: int, hi: int, check) -> int:
    while hi - lo > 3:
        m1 = lo + (hi - lo) // 3
        m2 = hi - (hi - lo) // 3
        if check(m1) < check(m2):
            hi = m2 - 1
        else:
            lo = m1 + 1
    best = lo
    for x in range(lo, hi + 1):
        if check(x) < check(best):
            best = x

def ternary_search(f, left, right, eps=1e-7):
    """
    Find x in [left, right] that minimizes f(x).
    f must be unimodal (convex in this case).
    eps = precision
    """
    while right - left > eps:
        m1 = left + (right - left) / 3
        m2 = right - (right - left) / 3
        
        if f(m1) < f(m2):
            right = m2   # min is in [left, m2]
        else:
            left = m1    # min is in [m1, right]
    
    return (left + right) / 2  # approximate min point
