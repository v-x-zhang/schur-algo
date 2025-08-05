"""
Exact Schur polynomial computations for validation against theoretical weights.
Based on the Jacobi-Trudi identity and determinant formulas.
"""

import numpy as np
from itertools import combinations_with_replacement, permutations
from math import factorial
from collections import defaultdict
import sympy as sp
from sympy import symbols, expand, Poly

def elementary_symmetric_polynomial(variables, k):
    """
    Compute the k-th elementary symmetric polynomial e_k(x1, x2, ..., xn).
    e_k = sum of all products of k distinct variables.
    """
    if k == 0:
        return 1
    if k > len(variables) or k < 0:
        return 0
    
    n = len(variables)
    if k == 1:
        return sum(variables)
    
    # Use recursive formula: e_k(x1,...,xn) = e_k(x1,...,x_{n-1}) + x_n * e_{k-1}(x1,...,x_{n-1})
    if n == 1:
        return variables[0] if k == 1 else 0
    
    # Split variables
    vars_without_last = variables[:-1]
    last_var = variables[-1]
    
    term1 = elementary_symmetric_polynomial(vars_without_last, k)
    term2 = last_var * elementary_symmetric_polynomial(vars_without_last, k-1)
    
    return term1 + term2

def complete_symmetric_polynomial(variables, k):
    """
    Compute the k-th complete symmetric polynomial h_k(x1, x2, ..., xn).
    h_k = sum of all monomials of degree k.
    """
    if k == 0:
        return 1
    if k < 0:
        return 0
    if len(variables) == 0:
        return 0
    
    # Use generating function approach for small cases
    if len(variables) == 1:
        return variables[0]**k
    
    # Use Newton's identities: h_k = (1/k) * sum_{i=1}^k p_i * h_{k-i}
    # where p_i are power sums. For computational efficiency, use recurrence:
    # h_k(x1,...,xn) = h_k(x1,...,x_{n-1}) + x_n * h_{k-1}(x1,...,xn)
    
    vars_without_last = variables[:-1]
    last_var = variables[-1]
    
    term1 = complete_symmetric_polynomial(vars_without_last, k)
    term2 = last_var * complete_symmetric_polynomial(variables, k-1)
    
    return term1 + term2

def schur_polynomial_determinant(partition, variables):
    """
    Compute Schur polynomial s_λ(x1, ..., xn) using the Jacobi-Trudi determinant formula:
    s_λ = det(h_{λ_i - i + j})_{1 ≤ i,j ≤ ℓ(λ)}
    
    where h_k are complete symmetric polynomials and ℓ(λ) is the length of λ.
    """
    if not partition:
        return 1  # s_∅ = 1
    
    if not variables:
        return 0  # No variables to work with
    
    # Ensure partition is in decreasing order
    partition = tuple(sorted(partition, reverse=True))
    length = len(partition)
    
    # Create the matrix for the determinant
    # Entry (i,j) is h_{λ_i - i + j}
    matrix = []
    for i in range(length):
        row = []
        for j in range(length):
            k = partition[i] - i + j  # This is λ_i - i + j (using 0-based indexing)
            h_k = complete_symmetric_polynomial(variables, k)
            row.append(h_k)
        matrix.append(row)
    
    # Convert to sympy matrix and compute determinant
    if all(isinstance(entry, (int, float)) for row in matrix for entry in row):
        # All entries are numeric
        matrix_np = np.array(matrix, dtype=float)
        return np.linalg.det(matrix_np)
    else:
        # Contains symbolic expressions
        matrix_sp = sp.Matrix(matrix)
        det = matrix_sp.det()
        return det

def skew_schur_polynomial(lambda_partition, mu_partition, variables):
    """
    Compute skew Schur polynomial s_{λ/μ}(x1, ..., xn) using the determinant formula:
    s_{λ/μ} = det(h_{λ_i - μ_j - i + j})_{1 ≤ i,j ≤ max(ℓ(λ), ℓ(μ))}
    
    where we pad partitions with zeros as needed.
    """
    if not lambda_partition:
        return 1 if not mu_partition else 0
    
    if not mu_partition:
        return schur_polynomial_determinant(lambda_partition, variables)
    
    if not variables:
        return 0
    
    # Ensure partitions are in decreasing order
    lambda_partition = tuple(sorted(lambda_partition, reverse=True))
    mu_partition = tuple(sorted(mu_partition, reverse=True))
    
    # Check if μ ⊆ λ (mu is contained in lambda)
    # Pad mu with zeros to match length of lambda
    max_length = max(len(lambda_partition), len(mu_partition))
    lambda_padded = list(lambda_partition) + [0] * (max_length - len(lambda_partition))
    mu_padded = list(mu_partition) + [0] * (max_length - len(mu_partition))
    
    # Check containment condition
    for i in range(max_length):
        if lambda_padded[i] < mu_padded[i]:
            return 0  # μ is not contained in λ
    
    # Create the matrix for the determinant
    matrix = []
    for i in range(max_length):
        row = []
        for j in range(max_length):
            k = lambda_padded[i] - mu_padded[j] - i + j
            h_k = complete_symmetric_polynomial(variables, k)
            row.append(h_k)
        matrix.append(row)
    
    # Convert to sympy matrix and compute determinant
    if all(isinstance(entry, (int, float)) for row in matrix for entry in row):
        # All entries are numeric
        matrix_np = np.array(matrix, dtype=float)
        return np.linalg.det(matrix_np)
    else:
        # Contains symbolic expressions
        matrix_sp = sp.Matrix(matrix)
        det = matrix_sp.det()
        return det

def exact_schur_weight(partition_sequence, X, Y):
    """
    Compute the exact theoretical weight P_{X,Y}(λ1, ..., λM) using the formula:
    
    P_{X,Y}(λ1, ..., λM) = ∏_{i=1}^M ∏_{j=1}^N (1 - x_i * y_j) * 
                           ∏_{i=1}^M s_{λi/λ_{i-1}}(x_i) * s_{λM}(y_1, ..., y_N)
    
    where λ0 = ∅ (empty partition) and we use exact Schur polynomial computations.
    """
    M = len(X)
    N = len(Y)
    
    # First factor: ∏_{i=1}^M ∏_{j=1}^N (1 - x_i * y_j)
    normalization = 1.0
    for i in range(M):
        for j in range(N):
            normalization *= (1 - X[i] * Y[j])
    
    # Second factor: ∏_{i=1}^M s_{λi/λ_{i-1}}(x_i)
    skew_schur_product = 1.0
    prev_partition = ()  # λ_{i-1} starts as empty
    
    for i, current_partition in enumerate(partition_sequence):
        # Compute skew Schur function s_{λi/λ_{i-1}}(x_i)
        skew_value = skew_schur_polynomial(current_partition, prev_partition, [X[i]])
        skew_schur_product *= float(skew_value) if hasattr(skew_value, 'evalf') else skew_value
        prev_partition = current_partition
    
    # Third factor: s_{λM}(y_1, ..., y_N)
    final_partition = partition_sequence[-1] if partition_sequence else ()
    final_schur = schur_polynomial_determinant(final_partition, Y)
    final_schur_value = float(final_schur) if hasattr(final_schur, 'evalf') else final_schur
    
    weight = normalization * skew_schur_product * final_schur_value
    return weight

def enumerate_partitions_bounded(max_parts, max_size, max_weight=None):
    """
    Enumerate all partitions with at most max_parts parts, each of size at most max_size,
    and optionally with total weight at most max_weight.
    """
    partitions = [()]  # Start with empty partition
    
    for weight in range(1, (max_weight or max_parts * max_size) + 1):
        # Generate all partitions of given weight
        for num_parts in range(1, min(max_parts, weight) + 1):
            # Generate partitions with exactly num_parts parts
            for parts in generate_partitions_of_weight(weight, num_parts, max_size):
                partitions.append(parts)
    
    return partitions

def generate_partitions_of_weight(weight, num_parts, max_part_size):
    """
    Generate all partitions of a given weight with exactly num_parts parts,
    each part at most max_part_size.
    """
    if num_parts == 1:
        if weight <= max_part_size:
            yield (weight,)
        return
    
    # Try all possible values for the largest part
    for largest in range(min(weight, max_part_size), 0, -1):
        remaining_weight = weight - largest
        remaining_parts = num_parts - 1
        
        if remaining_weight == 0:
            if remaining_parts == 0:
                yield (largest,)
            continue
        
        # Generate partitions for the remaining weight
        for sub_partition in generate_partitions_of_weight(remaining_weight, remaining_parts, largest):
            # Combine with current largest part
            full_partition = (largest,) + sub_partition
            # Ensure decreasing order
            yield tuple(sorted(full_partition, reverse=True))

def validate_against_exact_schur_weights(X, Y, sim_count=5000, max_parts=2, max_size=2):
    """
    Validate samplers against exact theoretical Schur process weights.
    """
    print("Validation against EXACT Schur process weights")
    print(f"Parameters: X={X}, Y={Y}")
    print(f"Simulations: {sim_count}")
    print(f"Max partition parts: {max_parts}, Max part size: {max_size}")
    
    from schur_sampler import sample_push_block_grid, sample_rsk_grid
    from collections import Counter
    import itertools
    
    M = len(X)
    
    # Generate all possible partition sequences (bounded)
    small_partitions = enumerate_partitions_bounded(max_parts, max_size, max_parts * max_size)
    partition_sequences = list(itertools.product(small_partitions, repeat=M))
    
    # Filter sequences that are too large (computational limit)
    if len(partition_sequences) > 1000:
        print(f"Warning: {len(partition_sequences)} sequences is too many. Limiting to first 500.")
        partition_sequences = partition_sequences[:500]
    
    print(f"Analyzing {len(partition_sequences)} possible partition sequences...")
    
    # Collect empirical frequencies
    pb_frequencies = defaultdict(int)
    rsk_frequencies = defaultdict(int)
    
    for sim in range(sim_count):
        if sim % (sim_count // 10) == 0 and sim > 0:
            print(f"Progress: {sim}/{sim_count}")
        
        try:
            # Sample grids
            pb_grid = sample_push_block_grid(X, Y)
            rsk_grid = sample_rsk_grid(X, Y)
            
            # Extract final row (the partition sequence we're interested in)
            pb_sequence = tuple(tuple(pb_grid[-1][m]._parts) for m in range(1, M + 1))
            rsk_sequence = tuple(tuple(rsk_grid[-1][m]._parts) for m in range(1, M + 1))
            
            # Count all sequences (not just the bounded ones)
            pb_frequencies[pb_sequence] += 1
            rsk_frequencies[rsk_sequence] += 1
            
        except Exception as e:
            print(f"Error in simulation {sim}: {e}")
            continue
    
    # Compute exact theoretical weights
    print("Computing exact theoretical weights...")
    theoretical_weights = {}
    total_theoretical_weight = 0
    
    for i, seq in enumerate(partition_sequences):
        if i % 100 == 0 and i > 0:
            print(f"  Computing weight {i}/{len(partition_sequences)}")
        
        try:
            weight = exact_schur_weight(seq, X, Y)
            theoretical_weights[seq] = weight
            total_theoretical_weight += weight
        except Exception as e:
            print(f"Error computing weight for {seq}: {e}")
            theoretical_weights[seq] = 0
    
    # Normalize theoretical weights to probabilities
    if total_theoretical_weight > 0:
        for seq in theoretical_weights:
            theoretical_weights[seq] /= total_theoretical_weight
    
    # Compare with empirical frequencies
    print(f"\nComparison of partition sequences (exact Schur weights):")
    print("Sequence | Push-Block | RSK | Theoretical | PB Diff | RSK Diff")
    print("-" * 85)
    
    # Sort by theoretical weight (descending)
    sorted_sequences = sorted(theoretical_weights.items(), key=lambda x: x[1], reverse=True)
    
    total_pb_diff = 0
    total_rsk_diff = 0
    sequences_compared = 0
    total_empirical_pb = sum(pb_frequencies.values())
    total_empirical_rsk = sum(rsk_frequencies.values())
    
    for seq, theoretical_prob in sorted_sequences[:20]:  # Show top 20
        pb_prob = pb_frequencies[seq] / total_empirical_pb if total_empirical_pb > 0 else 0
        rsk_prob = rsk_frequencies[seq] / total_empirical_rsk if total_empirical_rsk > 0 else 0
        
        pb_diff = abs(pb_prob - theoretical_prob)
        rsk_diff = abs(rsk_prob - theoretical_prob)
        
        total_pb_diff += pb_diff
        total_rsk_diff += rsk_diff
        sequences_compared += 1
        
        # Format sequence for display
        seq_str = str(seq)
        if len(seq_str) > 15:
            seq_str = seq_str[:12] + "..."
        
        print(f"{seq_str:15s} | {pb_prob:10.6f} | {rsk_prob:7.6f} | {theoretical_prob:11.6f} | "
              f"{pb_diff:7.6f} | {rsk_diff:8.6f}")
    
    avg_pb_diff = total_pb_diff / sequences_compared if sequences_compared > 0 else 0
    avg_rsk_diff = total_rsk_diff / sequences_compared if sequences_compared > 0 else 0
    
    # Check how much of the empirical distribution is captured by our theoretical analysis
    captured_pb = sum(pb_frequencies[seq] for seq in theoretical_weights if seq in pb_frequencies)
    captured_rsk = sum(rsk_frequencies[seq] for seq in theoretical_weights if seq in rsk_frequencies)
    
    print(f"\nSummary:")
    print(f"Average difference from theoretical (Push-Block): {avg_pb_diff:.8f}")
    print(f"Average difference from theoretical (RSK): {avg_rsk_diff:.8f}")
    print(f"Sequences compared: {sequences_compared}")
    print(f"Total theoretical weight: {sum(theoretical_weights.values()):.8f}")
    print(f"Empirical coverage (Push-Block): {captured_pb/total_empirical_pb:.4f}")
    print(f"Empirical coverage (RSK): {captured_rsk/total_empirical_rsk:.4f}")
    
    return {
        'pb_frequencies': dict(pb_frequencies),
        'rsk_frequencies': dict(rsk_frequencies),
        'theoretical_weights': theoretical_weights,
        'avg_pb_difference': avg_pb_diff,
        'avg_rsk_difference': avg_rsk_diff,
        'sequences_compared': sequences_compared,
        'empirical_coverage_pb': captured_pb/total_empirical_pb if total_empirical_pb > 0 else 0,
        'empirical_coverage_rsk': captured_rsk/total_empirical_rsk if total_empirical_rsk > 0 else 0
    }

def test_schur_polynomials():
    """Test the Schur polynomial implementations with known examples."""
    print("Testing Schur polynomial implementations...")
    
    # Test elementary symmetric polynomials
    print("\nTesting elementary symmetric polynomials:")
    x = [1, 2, 3]
    e0 = elementary_symmetric_polynomial(x, 0)  # Should be 1
    e1 = elementary_symmetric_polynomial(x, 1)  # Should be 1+2+3 = 6
    e2 = elementary_symmetric_polynomial(x, 2)  # Should be 1*2 + 1*3 + 2*3 = 11
    e3 = elementary_symmetric_polynomial(x, 3)  # Should be 1*2*3 = 6
    print(f"e_0([1,2,3]) = {e0} (expected: 1)")
    print(f"e_1([1,2,3]) = {e1} (expected: 6)")
    print(f"e_2([1,2,3]) = {e2} (expected: 11)")
    print(f"e_3([1,2,3]) = {e3} (expected: 6)")
    
    # Test complete symmetric polynomials
    print("\nTesting complete symmetric polynomials:")
    h0 = complete_symmetric_polynomial(x, 0)  # Should be 1
    h1 = complete_symmetric_polynomial(x, 1)  # Should be 1+2+3 = 6
    h2 = complete_symmetric_polynomial(x, 2)  # Should be 1² + 2² + 3² + 1*2 + 1*3 + 2*3 = 1+4+9+2+3+6 = 25
    print(f"h_0([1,2,3]) = {h0} (expected: 1)")
    print(f"h_1([1,2,3]) = {h1} (expected: 6)")
    print(f"h_2([1,2,3]) = {h2} (should be positive)")
    
    # Test Schur polynomials for simple partitions
    print("\nTesting Schur polynomials:")
    s_empty = schur_polynomial_determinant((), x)  # Should be 1
    s_1 = schur_polynomial_determinant((1,), x)   # Should be h_1 = 6
    s_2 = schur_polynomial_determinant((2,), x)   # Should be h_2
    print(f"s_∅([1,2,3]) = {s_empty} (expected: 1)")
    print(f"s_(1)([1,2,3]) = {s_1} (expected: 6)")
    print(f"s_(2)([1,2,3]) = {s_2}")
    
    # Test skew Schur polynomials
    print("\nTesting skew Schur polynomials:")
    s_2_1 = skew_schur_polynomial((2,), (1,), x)  # s_(2)/(1)
    s_1_empty = skew_schur_polynomial((1,), (), x)  # Should equal s_(1) = 6
    print(f"s_(2)/(1)([1,2,3]) = {s_2_1}")
    print(f"s_(1)/∅([1,2,3]) = {s_1_empty} (expected: 6)")

if __name__ == "__main__":
    test_schur_polynomials()
