
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
        if lambda_i == 0:
            break
            
        if lambda_i > 0:  # Only add non-zero parts
            lambda_parts.append(lambda_i)
        
        # Set CARRY := min(μi, νi) − ρi and i := i + 1
        carry = min(mu_i, nu_i) - rho_i
        i += 1
        
        # Safety check to prevent infinite loops
        if i > 100000000:  # Reasonable upper bound
            print("Warning: Exceeded maximum iterations in rsk_algorithm_f1, breaking to avoid infinite loop.")
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
            rho = λ[n][m-1] if m > 1 else PartitionWrapper([])  # left above
            
            q = X[m-1] * Y[n]  # probability parameter
            
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


