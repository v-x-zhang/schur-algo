#!/usr/bin/env python3
"""
Find ONE specific exclusive grid configuration and verify it with extensive sampling.
"""

from schur_sampler import sample_push_block_grid, sample_rsk_grid
import numpy as np
from collections import Counter
from tqdm import tqdm

def grid_to_tuple(grid):
    """Convert grid to hashable tuple representation."""
    return tuple(
        tuple(tuple(partition._parts) for partition in row) 
        for row in grid
    )

def print_grid_from_tuple(grid_tuple, title="Grid"):
    """Print a grid from its tuple representation."""
    print(f"\n{title}:")
    print("=" * (len(title) + 1))
    
    for i, row_tuple in enumerate(grid_tuple):
        row_str = []
        for j, partition_tuple in enumerate(row_tuple):
            # Convert partition tuple to a compact string
            if not partition_tuple:
                part_str = "∅"
            elif len(partition_tuple) == 1:
                part_str = str(partition_tuple[0])
            else:
                part_str = f"({','.join(map(str, partition_tuple))})"
            row_str.append(f"{part_str:>8}")
        print(f"Row {i}: [" + " ".join(row_str) + "]")

def find_one_exclusive_grid(X, Y, num_samples=5000):
    """
    Find exactly one grid configuration that is exclusive to one algorithm.
    
    Returns:
    - exclusive_grid: The exclusive grid configuration (tuple)
    - algorithm: Which algorithm it belongs to ('push_block' or 'rsk')
    """
    print(f"Searching for one exclusive grid with X={X}, Y={Y}")
    print(f"Running {num_samples} samples for each algorithm...")
    
    push_grids = set()
    rsk_grids = set()
    
    # Sample from push-block algorithm
    print("Sampling Push-Block grids...")
    for _ in tqdm(range(num_samples), desc="Push-Block"):
        grid = sample_push_block_grid(X, Y)
        grid_tuple = grid_to_tuple(grid)
        push_grids.add(grid_tuple)
    
    # Sample from RSK algorithm  
    print("Sampling RSK grids...")
    for _ in tqdm(range(num_samples), desc="RSK"):
        grid = sample_rsk_grid(X, Y)
        grid_tuple = grid_to_tuple(grid)
        rsk_grids.add(grid_tuple)
    
    # Find exclusive grids
    push_only = push_grids - rsk_grids
    rsk_only = rsk_grids - push_grids
    
    print(f"\nInitial search results:")
    print(f"Push-Block exclusive grids: {len(push_only)}")
    print(f"RSK exclusive grids: {len(rsk_only)}")
    
    # Pick one exclusive grid (prefer one with interesting structure)
    if push_only:
        # Look for a grid with multi-part partitions if available
        for grid in push_only:
            has_multipart = any(
                len(partition_tuple) > 1 
                for row_tuple in grid 
                for partition_tuple in row_tuple
            )
            if has_multipart:
                print(f"\nSelected Push-Block exclusive grid with multi-part partitions:")
                print_grid_from_tuple(grid, "Selected Push-Block Exclusive Grid")
                return grid, 'push_block'
        
        # If no multi-part, just take the first one
        selected_grid = list(push_only)[0]
        print(f"\nSelected Push-Block exclusive grid:")
        print_grid_from_tuple(selected_grid, "Selected Push-Block Exclusive Grid")
        return selected_grid, 'push_block'
    
    elif rsk_only:
        # Look for a grid with multi-part partitions if available
        for grid in rsk_only:
            has_multipart = any(
                len(partition_tuple) > 1 
                for row_tuple in grid 
                for partition_tuple in row_tuple
            )
            if has_multipart:
                print(f"\nSelected RSK exclusive grid with multi-part partitions:")
                print_grid_from_tuple(grid, "Selected RSK Exclusive Grid")
                return grid, 'rsk'
        
        # If no multi-part, just take the first one
        selected_grid = list(rsk_only)[0]
        print(f"\nSelected RSK exclusive grid:")
        print_grid_from_tuple(selected_grid, "Selected RSK Exclusive Grid")
        return selected_grid, 'rsk'
    
    else:
        print(f"\n❌ No exclusive grids found with this parameter set!")
        return None, None

def verify_exclusivity(X, Y, target_grid, expected_algorithm, num_simulations=10000):
    """
    Verify that the target grid only appears in the expected algorithm with extensive sampling.
    
    Args:
    - target_grid: The grid configuration to verify (tuple)
    - expected_algorithm: 'push_block' or 'rsk'
    - num_simulations: Number of simulations to run for verification
    """
    print(f"\n{'='*80}")
    print(f"VERIFICATION PHASE")
    print(f"{'='*80}")
    print(f"Target grid should ONLY appear in: {expected_algorithm.upper()}")
    print(f"Running {num_simulations} simulations for each algorithm...")
    
    push_count = 0
    rsk_count = 0
    
    # Count occurrences in Push-Block algorithm
    print(f"\nTesting Push-Block algorithm...")
    for _ in tqdm(range(num_simulations), desc="Push-Block Verification"):
        grid = sample_push_block_grid(X, Y)
        grid_tuple = grid_to_tuple(grid)
        if grid_tuple == target_grid:
            push_count += 1
    
    # Count occurrences in RSK algorithm
    print(f"\nTesting RSK algorithm...")
    for _ in tqdm(range(num_simulations), desc="RSK Verification"):
        grid = sample_rsk_grid(X, Y)
        grid_tuple = grid_to_tuple(grid)
        if grid_tuple == target_grid:
            rsk_count += 1
    
    # Results
    print(f"\n{'='*80}")
    print(f"VERIFICATION RESULTS")
    print(f"{'='*80}")
    print(f"Target grid configuration:")
    print_grid_from_tuple(target_grid, "Target Grid")
    
    print(f"\nFrequency results from {num_simulations} simulations each:")
    print(f"Push-Block occurrences: {push_count} ({push_count/num_simulations*100:.4f}%)")
    print(f"RSK occurrences: {rsk_count} ({rsk_count/num_simulations*100:.4f}%)")
    
    # Verification
    if expected_algorithm == 'push_block':
        if push_count > 0 and rsk_count == 0:
            print(f"\n✅ VERIFIED: Grid is EXCLUSIVE to Push-Block algorithm!")
            print(f"   - Appeared {push_count} times in Push-Block")
            print(f"   - Never appeared in RSK (0 times)")
        elif push_count > 0 and rsk_count > 0:
            print(f"\n❌ FAILED: Grid appears in BOTH algorithms!")
            print(f"   - This grid is NOT exclusive")
        elif push_count == 0:
            print(f"\n❌ FAILED: Grid doesn't appear in Push-Block either!")
            print(f"   - This might be a very rare configuration")
        else:
            print(f"\n⚠️  UNEXPECTED: Grid appears in RSK but not Push-Block")
    
    elif expected_algorithm == 'rsk':
        if rsk_count > 0 and push_count == 0:
            print(f"\n✅ VERIFIED: Grid is EXCLUSIVE to RSK algorithm!")
            print(f"   - Appeared {rsk_count} times in RSK")
            print(f"   - Never appeared in Push-Block (0 times)")
        elif push_count > 0 and rsk_count > 0:
            print(f"\n❌ FAILED: Grid appears in BOTH algorithms!")
            print(f"   - This grid is NOT exclusive")
        elif rsk_count == 0:
            print(f"\n❌ FAILED: Grid doesn't appear in RSK either!")
            print(f"   - This might be a very rare configuration")
        else:
            print(f"\n⚠️  UNEXPECTED: Grid appears in Push-Block but not RSK")
    
    return push_count, rsk_count

def main():
    """Main function to find and verify one exclusive grid configuration."""
    
    # Use parameters that showed good exclusivity
    X = [0.1, 0.2]
    Y = [0.7, 0.8]
    
    print("FINDING ONE EXCLUSIVE GRID CONFIGURATION")
    print("="*80)
    
    # Step 1: Find one exclusive grid
    target_grid, algorithm = find_one_exclusive_grid(X, Y, num_samples=5000)
    
    if target_grid is None:
        print("❌ Could not find an exclusive grid. Trying different parameters...")
        # Try alternative parameters
        X = [0.2, 0.6]
        Y = [0.2, 0.6]
        target_grid, algorithm = find_one_exclusive_grid(X, Y, num_samples=5000)
    
    if target_grid is None:
        print("❌ Could not find any exclusive grid configurations!")
        return
    
    # Step 2: Verify with extensive sampling
    push_count, rsk_count = verify_exclusivity(X, Y, target_grid, algorithm, num_simulations=10000)
    
    # Step 3: Final summary
    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"Parameters: X={X}, Y={Y}")
    print(f"Expected exclusive algorithm: {algorithm.upper()}")
    print(f"Verification with 10,000 simulations each:")
    print(f"  Push-Block: {push_count} occurrences")
    print(f"  RSK: {rsk_count} occurrences")
    
    if (algorithm == 'push_block' and push_count > 0 and rsk_count == 0) or \
       (algorithm == 'rsk' and rsk_count > 0 and push_count == 0):
        print(f"\n🎯 SUCCESS: Found and verified an exclusive grid configuration!")
    else:
        print(f"\n❌ Verification failed - grid is not truly exclusive")

if __name__ == "__main__":
    main()
