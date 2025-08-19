#!/usr/bin/env python3
"""
Show detailed grid structures for debugging and visualization.
Find grid configurations exclusive to each sampler.
"""

from schur_sampler import sample_push_block_grid, sample_rsk_grid
import numpy as np
from collections import defaultdict
from tqdm import tqdm

def grid_to_tuple(grid):
    """Convert grid to hashable tuple representation."""
    return tuple(
        tuple(tuple(partition._parts) for partition in row) 
        for row in grid
    )

def print_grid_structure(grid, title="Grid"):
    """Print a readable representation of a grid with partitions."""
    print(f"\n{title}:")
    print("=" * (len(title) + 1))
    
    for i, row in enumerate(grid):
        row_str = []
        for j, partition in enumerate(row):
            # Convert partition to a compact string
            parts = list(partition._parts) if partition._parts else []
            if not parts:
                part_str = "∅"
            elif len(parts) == 1:
                part_str = str(parts[0])
            else:
                part_str = f"({','.join(map(str, parts))})"
            row_str.append(f"{part_str:>8}")
        print(f"Row {i}: [" + " ".join(row_str) + "]")

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

def find_exclusive_grids(X, Y, num_samples=10000):
    """
    Find grid configurations that are exclusive to each sampler.
    
    Returns:
    - push_only: Set of grids that only appear in push-block sampling
    - rsk_only: Set of grids that only appear in RSK sampling  
    - common: Set of grids that appear in both
    """
    print(f"Searching for exclusive grid configurations with X={X}, Y={Y}")
    print(f"Grid size: {len(Y)+1} x {len(X)+1}")
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
    
    # Find exclusive and common grids
    push_only = push_grids - rsk_grids
    rsk_only = rsk_grids - push_grids
    common = push_grids & rsk_grids
    
    print(f"\nResults:")
    print(f"Push-Block unique grids: {len(push_grids)}")
    print(f"RSK unique grids: {len(rsk_grids)}")
    print(f"Common grids: {len(common)}")
    print(f"Push-Block EXCLUSIVE grids: {len(push_only)}")
    print(f"RSK EXCLUSIVE grids: {len(rsk_only)}")
    
    return push_only, rsk_only, common

def analyze_exclusive_grids(push_only, rsk_only, max_display=5):
    """Analyze and display the exclusive grid configurations."""
    
    if push_only:
        print(f"\n{'='*60}")
        print(f"PUSH-BLOCK EXCLUSIVE GRIDS (showing up to {max_display}):")
        print(f"{'='*60}")
        
        for i, grid_tuple in enumerate(list(push_only)[:max_display]):
            print(f"\nPush-Block Exclusive Grid #{i+1}:")
            print_grid_from_tuple(grid_tuple, f"Push-Block Exclusive #{i+1}")
            
            # Analyze characteristics
            analyze_grid_characteristics(grid_tuple, "Push-Block Exclusive")
    else:
        print(f"\n❌ No Push-Block exclusive grids found!")
    
    if rsk_only:
        print(f"\n{'='*60}")
        print(f"RSK EXCLUSIVE GRIDS (showing up to {max_display}):")
        print(f"{'='*60}")
        
        for i, grid_tuple in enumerate(list(rsk_only)[:max_display]):
            print(f"\nRSK Exclusive Grid #{i+1}:")
            print_grid_from_tuple(grid_tuple, f"RSK Exclusive #{i+1}")
            
            # Analyze characteristics
            analyze_grid_characteristics(grid_tuple, "RSK Exclusive")
    else:
        print(f"\n❌ No RSK exclusive grids found!")

def analyze_grid_characteristics(grid_tuple, label):
    """Analyze and print characteristics of a grid configuration."""
    total_parts = 0
    max_partition_size = 0
    non_empty_positions = 0
    
    for i, row_tuple in enumerate(grid_tuple):
        for j, partition_tuple in enumerate(row_tuple):
            if partition_tuple:  # Non-empty partition
                non_empty_positions += 1
                total_parts += len(partition_tuple)
                max_partition_size = max(max_partition_size, max(partition_tuple) if partition_tuple else 0)
    
    print(f"  {label} characteristics:")
    print(f"    - Non-empty positions: {non_empty_positions}")
    print(f"    - Total partition parts: {total_parts}")
    print(f"    - Maximum partition value: {max_partition_size}")
    
    # Check for specific patterns
    patterns = []
    if non_empty_positions == 1:
        patterns.append("Single non-empty position")
    if total_parts == non_empty_positions:
        patterns.append("All partitions are singletons")
    if any(len(partition_tuple) > 1 for row_tuple in grid_tuple for partition_tuple in row_tuple):
        patterns.append("Contains multi-part partitions")
    
    if patterns:
        print(f"    - Patterns: {', '.join(patterns)}")

def compare_specific_grids(X, Y, num_samples=3):
    """Generate and compare specific grid examples."""
    print(f"Comparing specific grid examples with X={X}, Y={Y}")
    print(f"Grid size: {len(Y)+1} x {len(X)+1}")
    
    for i in range(num_samples):
        print(f"\n{'='*60}")
        print(f"SAMPLE {i+1}")
        print(f"{'='*60}")
        
        # Generate grids
        push_grid = sample_push_block_grid(X, Y)
        rsk_grid = sample_rsk_grid(X, Y)
        
        # Print both grids
        print_grid_structure(push_grid, "Push-Block Grid")
        print_grid_structure(rsk_grid, "RSK Grid")
        
        # Check if they're the same
        def grids_equal(g1, g2):
            if len(g1) != len(g2) or len(g1[0]) != len(g2[0]):
                return False
            for i in range(len(g1)):
                for j in range(len(g1[0])):
                    if tuple(g1[i][j]._parts) != tuple(g2[i][j]._parts):
                        return False
            return True
        
        if grids_equal(push_grid, rsk_grid):
            print("\n✓ IDENTICAL GRIDS!")
        else:
            print("\n✗ Different grids")
            
            # Show differences
            print("\nDifferences:")
            for i in range(len(push_grid)):
                for j in range(len(push_grid[0])):
                    push_parts = tuple(push_grid[i][j]._parts)
                    rsk_parts = tuple(rsk_grid[i][j]._parts)
                    if push_parts != rsk_parts:
                        print(f"  Position ({i},{j}): Push-Block={push_parts}, RSK={rsk_parts}")

if __name__ == "__main__":
    # Test with 3x3 grid (default parameters)
    X = [0.3, 0.4]
    Y = [0.3, 0.4]
    
    print("SEARCHING FOR EXCLUSIVE GRID CONFIGURATIONS")
    print("="*60)
    
    # Find exclusive grids
    push_only, rsk_only, common = find_exclusive_grids(X, Y, num_samples=5000)
    
    # Analyze and display the exclusive grids
    analyze_exclusive_grids(push_only, rsk_only, max_display=3)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    if push_only:
        print(f"✓ Found {len(push_only)} Push-Block exclusive grid(s)!")
    if rsk_only:
        print(f"✓ Found {len(rsk_only)} RSK exclusive grid(s)!")
    if not push_only and not rsk_only:
        print("❌ No exclusive grids found - algorithms may be equivalent for this parameter set")
    
    print(f"\nJaccard similarity: {len(common) / (len(push_only) + len(rsk_only) + len(common)):.4f}")
    
    # Also show a few random examples for comparison
    print(f"\n{'='*60}")
    print("RANDOM SAMPLE COMPARISON")
    print(f"{'='*60}")
    compare_specific_grids(X, Y, num_samples=2)
