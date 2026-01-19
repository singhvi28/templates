import random
import sys
import time

# ==========================================
# 1. YOUR OPTIMIZED SOLUTION
# ==========================================
def solution(input_str):
    try:
        # Example: Replace this with your actual logic
        lines = input_str.splitlines()
        if not lines: return ""
        return lines[0] # Dummy return
    except Exception as e:
        return f"RUNTIME_ERROR: {e}"

# ==========================================
# 2. YOUR BRUTE FORCE SOLUTION
# ==========================================
def brute(input_str):
    try:
        # Example: Replace this with your simple/slow logic
        lines = input_str.splitlines()
        if not lines: return ""
        return lines[0] # Dummy return
    except Exception as e:
        return f"RUNTIME_ERROR: {e}"

# ==========================================
# 3. GENERATOR HELPERS
# ==========================================

def gen_string(length, alphabet="abc"):
    return "".join(random.choice(alphabet) for _ in range(length))

def gen_tree(n):
    """Generates a random tree using the random parent method."""
    edges = []
    for i in range(2, n + 1):
        u = i
        v = random.randint(1, i - 1)
        edges.append((u, v))
    random.shuffle(edges)
    return edges

def gen_graph(n, m, directed=False):
    """Generates a connected graph with N nodes and M edges."""
    if m < n - 1: m = n - 1
    edges = set()
    # Ensure connectivity
    for i in range(2, n + 1):
        u, v = i, random.randint(1, i - 1)
        edges.add((u, v) if directed else tuple(sorted((u, v))))
    # Fill remaining edges
    while len(edges) < m:
        u, v = random.sample(range(1, n + 1), 2)
        edges.add((u, v) if directed else tuple(sorted((u, v))))
    return list(edges)

# ==========================================
# 4. TEST CASE CONFIGURATION (Tweak this!)
# ==========================================
def generate_test_case():
    # --- Example A: Array/Numeric ---
    # n = random.randint(1, 10)
    # arr = [random.randint(1, 100) for _ in range(n)]
    # return f"{n}\n" + " ".join(map(str, arr))

    # --- Example B: Strings ---
    # s = gen_string(random.randint(1, 10), "ab")
    # return s

    # --- Example C: Trees ---
    n = random.randint(2, 6)
    edges = gen_tree(n)
    res = [str(n)]
    for u, v in edges:
        res.append(f"{u} {v}")
    return "\n".join(res)

# ==========================================
# 5. ENGINE
# ==========================================
def run_stress_test(max_tests=5000):
    print(f"🚀 Starting stress test...")
    
    for i in range(1, max_tests + 1):
        test_input = generate_test_case()
        
        # Execute and time the solution
        start_time = time.perf_counter()
        sol_out = str(solution(test_input)).strip()
        end_time = time.perf_counter()
        
        brute_out = str(brute(test_input)).strip()
        
        # Check for discrepancies
        if sol_out != brute_out:
            print(f"\n❌ FAILED ON TEST {i}")
            print("--- INPUT ---")
            print(test_input)
            print("--- SOLUTION OUTPUT ---")
            print(sol_out)
            print("--- BRUTE OUTPUT ---")
            print(brute_out)
            return

        # Optional: Check for TLE (e.g., > 1 second)
        if (end_time - start_time) > 1.0:
            print(f"\n⚠️ TLE ON TEST {i} ({end_time - start_time:.2f}s)")
            print("--- INPUT ---")
            print(test_input)
            return

        if i % 100 == 0:
            print(f"✅ {i} tests passed...", end='\r')
        else:
            print(f"Testing: {i}", end='\r')

    print(f"\n✨ All {max_tests} tests passed successfully!")

if __name__ == "__main__":
    run_stress_test()
