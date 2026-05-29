# Customization Checklist:
# __init__ & _build(): Set up your aggregate variables (sums, maxes, 
# frequency counters, etc.) and calculate them from self.vals.

# apply_lazy(val): Define how a block's aggregate and lazy tag change 
# when the entire block receives an update.

# push(): Define how a pending lazy tag is distributed to the individual 
# elements in self.vals before a partial update/query.

# update_element(idx, val): Define how a single element is modified during 
# a partial block update.

# query_full() & query_element(idx): Return the data needed to construct the final answer.

# range_query(l, r): Update the res identity variable (e.g., 0 for sums, 
# float('inf') for minimums) and the logic that combines the block results.

from math import isqrt

class Block:
    def __init__(self, vals):
        self.vals = vals
        # 1. Initialize lazy tag and aggregate data
        self.lazy = 0 
        self.aggregate = 0 
        self._build()

    def _build(self):
        """Recompute aggregate data from scratch (used after partial block updates)."""
        # Example: self.aggregate = sum(self.vals)
        pass

    def push(self):
        """Push the lazy tag down to individual elements and clear the lazy tag."""
        if not self.lazy: # Change condition based on your "null" lazy value
            return
        
        # Example for range addition:
        # for i in range(len(self.vals)):
        #     self.vals[i] += self.lazy
        
        self.lazy = 0 # Reset lazy tag

    def apply_lazy(self, val):
        """Apply an update to the entire block in O(1) time."""
        # Example for range addition:
        # self.lazy += val
        # self.aggregate += val * len(self.vals)
        pass

    def update_element(self, idx, val):
        """Apply an update to a single element at local index `idx`."""
        # Example for range addition:
        # self.vals[idx] += val
        pass

    def query_full(self):
        """Return the answer for this entire block in O(1) time."""
        # Example:
        # return self.aggregate
        pass

    def query_element(self, idx):
        """Return the actual value/contribution of a single element at local index `idx`."""
        # Example:
        # return self.vals[idx]
        pass


class SqrtDecomp:
    def __init__(self, arr):
        self.B = isqrt(len(arr)) + 1
        self.blocks = [Block(arr[i:i+self.B]) for i in range(0, len(arr), self.B)]

    def range_update(self, l, r, val):
        for bi, block in enumerate(self.blocks):
            start = bi * self.B
            end = start + len(block.vals)

            # 1. Block is completely outside the range
            if r < start or l >= end:
                continue

            # 2. Block is completely inside the range
            if l <= start and end - 1 <= r:
                block.apply_lazy(val)
                continue

            # 3. Block partially overlaps the range
            block.push() # Resolve pending lazy updates before manual edits
            
            L = max(l, start) - start
            R = min(r + 1, end) - start
            
            for i in range(L, R):
                block.update_element(i, val)
                
            block._build() # Rebuild the block's aggregate data

    def range_query(self, l, r):
        res = 0 # Replace with the identity element for your operation (e.g., float('inf') for min)
        
        for bi, block in enumerate(self.blocks):
            start = bi * self.B
            end = start + len(block.vals)

            # 1. Block is completely outside the range
            if r < start or l >= end:
                continue

            # 2. Block is completely inside the range
            if l <= start and end - 1 <= r:
                # Example: res += block.query_full()
                continue

            # 3. Block partially overlaps the range
            block.push() # Ensure accurate element-level reads
            
            L = max(l, start) - start
            R = min(r + 1, end) - start
            
            for i in range(L, R):
                # Example: res += block.query_element(i)
                pass
                
        return res