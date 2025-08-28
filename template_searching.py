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

def check(x: int) -> int:
    # Define your unimodal integer function here
    pass

def ternary_search(lo: int, hi: int) -> int:
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
