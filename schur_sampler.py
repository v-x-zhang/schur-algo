
from sympy.combinatorics.partitions import Partition
import numpy as np

class PartitionWrapper:
    """Wrapper for sympy Partition to provide part(i) and set_part(i, value) methods."""
    
    def __init__(self, parts=None):
        if parts is None:
            parts = []
        # Store parts in decreasing order, remove zeros
        self._parts = sorted([p for p in parts if p > 0], reverse=True)
    
    def part(self, i):
        """Get the i-th part (1-indexed as in the prompt)."""
        if i < 1 or i > len(self._parts):
            return 0
        return self._parts[i-1]
    
    def set_part(self, i, value):
        """Set the i-th part (1-indexed) to value."""
        # Extend parts list if necessary
        while len(self._parts) < i:
            self._parts.append(0)
        
        if i >= 1:
            self._parts[i-1] = value
        
        # Remove zeros and sort
        self._parts = sorted([p for p in self._parts if p > 0], reverse=True)
    
    def __iter__(self):
        return iter(self._parts)
    
    def __len__(self):
        return len(self._parts)
    
    def __str__(self):
        return str(self._parts)
    
    def __repr__(self):
        return f"PartitionWrapper({self._parts})"

def sample_push_block_grid(X, Y):
    """
    X: list of M floats (x1…xM), Y: list of N floats (y1…yN) with xi*yj < 1
    returns: (N+1)x(M+1) grid λ of sympy Partition objects
    """
    M = len(X)
    N = len(Y)
    
    # Validate input parameters
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            if x * y >= 1:
                raise ValueError(f"Constraint violated: X[{i}] * Y[{j}] = {x * y} >= 1")
    
    # Step 1: Create (N+1)×(M+1) array λ of Partition objects
    λ = [[None for _ in range(M + 1)] for _ in range(N + 1)]
    
    # Step 2: Initialization - set λ[0][m] = λ[n][0] = empty partition
    for m in range(M + 1):
        λ[0][m] = PartitionWrapper([])  # Empty partition for top row
    for n in range(N + 1):
        λ[n][0] = PartitionWrapper([])  # Empty partition for left column
    
    # Step 3: Fill the grid using the sampling algorithm
    for n in range(N):  # n in 0..N-1
        for m in range(1, M + 1):  # m in 1..M
            # let q = X[m-1] * Y[n]
            q = X[m-1] * Y[n]

            # Initialize the new partition
            λ[n+1][m] = PartitionWrapper([])            # for i in 1..min(m, n+1):
            for i in range(1, min(m, n+1) + 1):
                a_i = max(λ[n+1][m-1].part(i), λ[n][m].part(i))                
                # b_i = ∞ if i==1 else min( λ[n+1][m-1].part(i-1), λ[n][m].part(i-1) )

                if i == 1:
                    b_i = float('inf')
                else:
                    b_i = min(λ[n+1][m-1].part(i-1), λ[n][m].part(i-1))
                
                # Debug print
                # print(f"  Sampling λ[{n+1}][{m}].part({i}): a_i={a_i}, b_i={b_i}, q={q:.3f}")
                # print(f"  Left: λ[{n+1}][{m-1}] = {list(λ[n+1][m-1])}")
                # print(f"  Above: λ[{n}][{m}] = {list(λ[n][m])}")
                
                # draw Xi ~ p(·| a_i, b_i, q )
                Xi = sample_truncated_geometric_pmf(a_i, b_i, q)
                Xi = int(Xi)  # Ensure it's a regular Python int
                # print(f"    Sampled: {Xi}")
                
                # set λ[n+1][m].set_part(i, Xi)
                λ[n+1][m].set_part(i, Xi)
    
    # Step 4: Return λ as List[List[Partition]] of shape (N+1)×(M+1)
    return λ

def rsk_algorithm_f1(mu, nu, rho, m):
    """
    Algorithm F1: Forward map for RSK bijection.
    Given partitions μ, ν, ρ and integer m, compute partition λ.
    
    Args:
        mu, nu, rho: PartitionWrapper objects
        m: non-negative integer
    
    Returns:
        PartitionWrapper object representing λ
    """
    # Step 0: Set CARRY := m and i := 1
    carry = m
    i = 1
    lambda_parts = []
    
    while True:
        # Step 1: Set λi := max(μi, νi) + CARRY
        mu_i = mu.part(i)
        nu_i = nu.part(i)
        rho_i = rho.part(i)
        
        lambda_i = max(mu_i, nu_i) + carry
        
        # Modified Step 2: Continue even if λi = 0, but stop if both μi and νi are 0
        if lambda_i == 0 and mu_i == 0 and nu_i == 0:
            break
            
        if lambda_i > 0:  # Only add non-zero parts
            lambda_parts.append(lambda_i)
        
        # Set CARRY := min(μi, νi) − ρi and i := i + 1
        carry = min(mu_i, nu_i) - rho_i
        i += 1
        
        # Safety check to prevent infinite loops
        if i > 100:  # Reasonable upper bound
            break
    
    return PartitionWrapper(lambda_parts)


def sample_rsk_grid(X, Y):
    """
    Sample RSK grid using Algorithm F1.
    X: list of M floats (x1…xM), Y: list of N floats (y1…yN) with xi*yj < 1
    returns: (N+1)x(M+1) grid λ of sympy Partition objects
    """
    M = len(X)
    N = len(Y)
    
    # Validate input parameters
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            if x * y >= 1:
                raise ValueError(f"Constraint violated: X[{i}] * Y[{j}] = {x * y} >= 1")

    # Step 1: Create (N+1)×(M+1) array λ of Partition objects
    λ = [[None for _ in range(M + 1)] for _ in range(N + 1)]
    
    # Step 2: Initialization - set λ[0][m] = λ[n][0] = empty partition
    for m in range(M + 1):
        λ[0][m] = PartitionWrapper([])  # Empty partition for top row
    for n in range(N + 1):
        λ[n][0] = PartitionWrapper([])  # Empty partition for left column
    
    # Step 3: Fill the grid using RSK Algorithm F1
    for n in range(N):  # n in 0..N-1
        for m in range(1, M + 1):  # m in 1..M
            # Get neighboring partitions μ and ν
            mu = λ[n][m]      # above
            nu = λ[n+1][m-1]  # left
            
            # Sample ρ partition (interlacing with both μ and ν)
            rho_parts = []
            q = X[m-1] * Y[n]  # probability parameter
            
            # Sample parts of ρ such that ρ ⪯ μ and ρ ⪯ ν
            max_parts = max(len(mu._parts), len(nu._parts), 1)
            
            for i in range(1, max_parts + 1):
                mu_i = mu.part(i)
                nu_i = nu.part(i)
                
                # ρ must satisfy ρi ≤ min(μi, νi)
                upper_bound = min(mu_i, nu_i)
                
                if upper_bound > 0:
                    # Sample ρi from geometric distribution on [0, upper_bound]
                    rho_i = sample_truncated_geometric_pmf(0, upper_bound, q)
                    rho_i = int(rho_i)  # Ensure it's a regular Python int
                    if rho_i > 0:
                        rho_parts.append(rho_i)
            
            rho = PartitionWrapper(rho_parts)
            
            # Sample m (carry value)
            m_val = sample_truncated_geometric_pmf(0, 10, q)  # bounded to prevent explosion
            m_val = int(m_val)  # Ensure it's a regular Python int
            
            # Apply Algorithm F1
            λ[n+1][m] = rsk_algorithm_f1(mu, nu, rho, m_val)
    
    # Step 4: Return λ as List[List[Partition]] of shape (N+1)×(M+1)
    return λ




def sample_truncated_geometric_pmf(a, b, q):
    """
    Sample from p(x | a, b, q) = q**x / sum(q**y for y in range(a, b+1))
    for x ∈ [a..b], else 0.
    
    This is exactly Definition 3.3 from the paper.
    """
    if q == 0:
        return a
    
    if b == float('inf'):
        # Infinite upper bound - use shifted geometric distribution
        if q >= 1:
            raise ValueError("Parameter q must be < 1 for convergence with infinite upper bound")
        # Sample from geometric distribution starting at a
        return a + np.random.geometric(1 - q) - 1
    
    # Finite range [a, b]
    if a > b:
        raise ValueError(f"Invalid range: a={a} > b={b}")
    
    if a == b:
        return a
    
    # Compute probabilities: p(x | a, b, q) = q**x / sum(q**y for y in range(a, b+1))
    x_values = list(range(a, b + 1))
    unnormalized_probs = [q**x for x in x_values]
    
    # Normalize
    total = sum(unnormalized_probs)
    if total == 0:
        return a  # Fallback
    
    probs = [p / total for p in unnormalized_probs]
    
    # Sample using numpy
    try:
        choice = np.random.choice(x_values, p=probs)
        return choice
    except:
        # Fallback to minimum value if sampling fails
        return a


# Test function
def test_schur_process():
    """Test the Schur process sampling with small parameters."""
    
    print("Testing Schur process sampling algorithm (Dimitrov 2024)...")
    
    # First test with a very small example
    print("\n=== Small Test Case ===")
    X_small = [0.2, 0.3, 0.6, 0.2]
    Y_small = [0.6, 0.7, 0.2]
    
    print(f"X = {X_small}, Y = {Y_small}")
    print(f"Grid size will be: {len(Y_small)+1} × {len(X_small)+1}")
    
    np.random.seed(45)
    λ_small = sample_push_block_grid(X_small, Y_small)
    
    print("Sampled partition grid:")
    for n in range(len(λ_small)):
        row_str = []
        for m in range(len(λ_small[0])):
            partition_str = str(list(λ_small[n][m]))
            row_str.append(partition_str)
        print(f"Row {n}: {row_str}")
    
    # Test case with moderate parameters
    # print("\n=== Larger Test Case ===")
    # X = [0.6, 0.5, 0.4]
    # Y = [0.5, 0.6]
    
    # print(f"X = {X}")
    # print(f"Y = {Y}")
    
    # # Check constraints
    # max_product = max(x * y for x in X for y in Y)
    # print(f"Maximum X[i] * Y[j] = {max_product}")
    
    # if max_product >= 1:
    #     print("Warning: Constraint X[i] * Y[j] < 1 violated!")
    #     return
    
    # # Sample multiple times to show variability
    # for run in range(2):
    #     print(f"\n--- Run {run + 1} ---")
    #     np.random.seed(42 + run)  # Different seed each run
    #     λ = sample_schur_process(X, Y)
        
    #     print(f"Sampled partition grid (shape: {len(λ)} × {len(λ[0])}):")
    #     for n in range(len(λ)):
    #         row_str = []
    #         for m in range(len(λ[0])):
    #             partition_str = str(list(λ[n][m])) if λ[n][m] else "[]"
    #             row_str.append(partition_str)
    #         print(f"Row {n}: {row_str}")
    
    return λ_small


def test_rsk_process():
    """Test the RSK process sampling with small parameters."""
    
    print("Testing RSK process sampling algorithm...")
    
    # Test with a very small example
    print("\n=== RSK Small Test Case ===")
    X_small = [0.2, 0.3]
    Y_small = [0.4, 0.5]
    
    print(f"X = {X_small}, Y = {Y_small}")
    print(f"Grid size will be: {len(Y_small)+1} × {len(X_small)+1}")
    
    np.random.seed(42)
    λ_rsk = sample_rsk_grid(X_small, Y_small)
    
    print("RSK sampled partition grid:")
    for n in range(len(λ_rsk)):
        row_str = []
        for m in range(len(λ_rsk[0])):
            partition_str = str(list(λ_rsk[n][m]))
            row_str.append(partition_str)
        print(f"Row {n}: {row_str}")
    
    return λ_rsk


# ===== BORODIN ALGORITHM IMPLEMENTATION =====

import random
import math

class BorodinPartition:
    """Simple partition wrapper supporting 1-indexed access for Borodin's algorithm."""
    def __init__(self, parts):
        # parts should be a non-increasing list of non-neg ints
        self.parts = parts.copy()

    def part(self, i):
        """Return the i-th part (1-indexed), or 0 if i > length."""
        return self.parts[i-1] if 1 <= i <= len(self.parts) else 0

    def length(self):
        return len(self.parts)

    def __repr__(self):
        return f"BorodinPartition({self.parts})"


def sample_truncated_geometric_borodin(a, b, q):
    """
    Draw k ~ Geom(q) truncated to [a,b], i.e.
      P(k) ∝ q^k   for k=a,…,b.
    We ignore the (1-q) factor since it cancels in normalization.
    """
    # build unnormalized pmf
    pmf = [q**k for k in range(a, b+1)]
    total = sum(pmf)
    if total == 0:
        return a
    r = random.random() * total
    cum = 0.0
    for idx, p in enumerate(pmf, start=a):
        cum += p
        if r < cum:
            return idx
    return b


def sample_push_block_grid_borodin(X, Y):
    """
    Borodin's push–block sampler.

    X: list of M floats (x1…xM)
    Y: list of N floats (y1…yN), with x_i * y_j < 1
    returns: (N+1)x(M+1) grid of BorodinPartition objects
    """
    M, N = len(X), len(Y)

    # Validate input parameters
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            if x * y >= 1:
                raise ValueError(f"Constraint violated: X[{i}] * Y[{j}] = {x * y} >= 1")

    # initialize empty grid of partitions
    Λ = [[BorodinPartition([]) for _ in range(M+1)] for _ in range(N+1)]

    # main down–right traversal
    for n in range(N):           # rows 0..N-1
        for m in range(1, M+1):  # cols 1..M
            mu = Λ[n][m]         # above
            nu = Λ[n+1][m-1]     # left
            q  = X[m-1] * Y[n]   # geometric parameter

            # determine how many parts we need to sample
            K = max(mu.length(), nu.length()) + 1

            new_parts = []
            for i in range(1, K+1):
                a_i = max(mu.part(i), nu.part(i))
                if i == 1:
                    b_i = math.inf
                else:
                    b_i = min(mu.part(i-1), nu.part(i-1))

                # sample the i-th part from Geom(q) truncated to [a_i, b_i]
                if b_i == math.inf:
                    # Handle infinite upper bound case
                    if q >= 1:
                        ℓ = a_i
                    else:
                        # Use geometric distribution starting at a_i
                        ℓ = a_i + np.random.geometric(1 - q) - 1
                else:
                    ℓ = sample_truncated_geometric_borodin(a_i, int(b_i), q)
                
                if ℓ > 0:
                    new_parts.append(ℓ)

            Λ[n+1][m] = BorodinPartition(new_parts)

    return Λ


def test_borodin_process():
    """Test the Borodin process sampling with small parameters."""
    
    print("Testing Borodin process sampling algorithm...")
    
    # Test with a very small example
    print("\n=== Borodin Small Test Case ===")
    X_small = [0.3, 0.4, 0.2]
    Y_small = [0.5, 0.1]
    
    print(f"X = {X_small}, Y = {Y_small}")
    print(f"Grid size will be: {len(Y_small)+1} × {len(X_small)+1}")
    
    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    λ_borodin = sample_push_block_grid_borodin(X_small, Y_small)
    
    print("Borodin sampled partition grid:")
    for n in range(len(λ_borodin)):
        row_str = []
        for m in range(len(λ_borodin[0])):
            partition_str = str(λ_borodin[n][m].parts)
            row_str.append(partition_str)
        print(f"Row {n}: {row_str}")
    
    return λ_borodin


if __name__ == "__main__":
    test_schur_process()
    print("\n" + "="*50)
    test_rsk_process()
    print("\n" + "="*50)
    test_borodin_process()

