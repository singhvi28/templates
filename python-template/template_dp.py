from collections import deque

class ConvexHullTrick:
    def __init__(self):
        # Stores tuples of (m, c) representing the line y = mx + c
        self.lines = deque()

    def _is_redundant(self, l1, l2, l3):
        """
        Helper method to check if line l2 is rendered useless by line l3.
        Avoids division to prevent floating point precision issues.
        """
        m1, c1 = l1
        m2, c2 = l2
        m3, c3 = l3
        # Mathematically equivalent to: intersection(l1, l3) <= intersection(l1, l2)
        return (c3 - c1) * (m1 - m2) <= (c2 - c1) * (m1 - m3)

    def add(self, m, c):
        """
        Add the line y = mx + c to the hull.
        RULE: Slopes (m) must be added in strictly decreasing order.
        """
        l3 = (m, c)
        
        # Remove lines from the back that are no longer part of the optimal "envelope"
        while len(self.lines) >= 2 and self._is_redundant(self.lines[-2], self.lines[-1], l3):
            self.lines.pop()
            
        self.lines.append(l3)

    def query(self, x):
        """
        Get the minimum y value across all lines at a specific x.
        RULE: Queries (x) must be monotonically increasing.
        """
        if not self.lines:
            return float('inf') # Or whatever your default maximum should be
            
        # Since x is increasing, if line 0 is worse than line 1 at this x,
        # line 0 will NEVER be optimal for any future x. We can safely throw it away.
        while len(self.lines) >= 2:
            m1, c1 = self.lines[0]
            m2, c2 = self.lines[1]
            
            if m1 * x + c1 >= m2 * x + c2:
                self.lines.popleft()
            else:
                break
                
        # The line at the front of the deque is guaranteed to be the best for this x
        best_m, best_c = self.lines[0]
        return best_m * x + best_c

# # 1. Initialize the Black Box
# cht = ConvexHullTrick()

# # Let's say we have these lines (slopes are decreasing: 5, 3, 1)
# # y = 5x + 2
# # y = 3x + 10
# # y = 1x + 25

# # 2. Add lines (respecting the rule: slopes must be decreasing)
# cht.add(5, 2)
# cht.add(3, 10)
# cht.add(1, 25)

# # 3. Query minimums (respecting the rule: x must be increasing)
# print(cht.query(1))  # Output: 7  (from 5*1 + 2)
# print(cht.query(3))  # Output: 17 (from 5*3 + 2, wait! 3*3+10 is 19. So 5x+2 is still best)
# print(cht.query(5))  # Output: 25 (from 3*5 + 10)
# print(cht.query(8))  # Output: 33 (from 1*8 + 25)