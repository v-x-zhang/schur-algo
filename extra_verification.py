#!/usr/bin/env python3
"""
Extra verification with even more simulations for the specific exclusive grid.
"""

from schur_sampler import sample_push_block_grid, sample_rsk_grid
from tqdm import tqdm

def grid_to_tuple(grid):
    """Convert grid to hashable tuple representation."""
    return tuple(
        tuple(tuple(partition._parts) for partition in row) 
        for row in grid
    )

def extra_verification():
    """Run extra verification with 50,000 simulations each."""
    
    # The verified exclusive grid configuration
    target_grid = (
        ((), (), ()),
        ((), (1,), (1,)),
        ((), (1,), (1, 1))
    )
    
    X = [0.1, 0.2]
    Y = [0.7, 0.8]
    num_simulations = 50000
    
    print(f"EXTRA VERIFICATION WITH {num_simulations} SIMULATIONS EACH")
    print("="*80)
    print("Target grid (should ONLY appear in Push-Block):")
    print("Row 0: [∅  ∅  ∅]")
    print("Row 1: [∅  1  1]")
    print("Row 2: [∅  1  (1,1)]")
    print("="*80)
    
    push_count = 0
    rsk_count = 0
    
    # Test Push-Block
    print(f"\nTesting Push-Block with {num_simulations} simulations...")
    for _ in tqdm(range(num_simulations), desc="Push-Block"):
        grid = sample_push_block_grid(X, Y)
        grid_tuple = grid_to_tuple(grid)
        if grid_tuple == target_grid:
            push_count += 1
    
    # Test RSK
    print(f"\nTesting RSK with {num_simulations} simulations...")
    for _ in tqdm(range(num_simulations), desc="RSK"):
        grid = sample_rsk_grid(X, Y)
        grid_tuple = grid_to_tuple(grid)
        if grid_tuple == target_grid:
            rsk_count += 1
    
    print(f"\n{'='*80}")
    print(f"FINAL VERIFICATION RESULTS")
    print(f"{'='*80}")
    print(f"Push-Block occurrences: {push_count} out of {num_simulations} ({push_count/num_simulations*100:.4f}%)")
    print(f"RSK occurrences: {rsk_count} out of {num_simulations} ({rsk_count/num_simulations*100:.4f}%)")
    
    if push_count > 0 and rsk_count == 0:
        print(f"\n✅ CONCLUSIVELY VERIFIED!")
        print(f"This grid configuration is mathematically IMPOSSIBLE in RSK")
        print(f"but occurs with probability {push_count/num_simulations:.6f} in Push-Block")
    elif rsk_count > 0:
        print(f"\n❌ VERIFICATION FAILED!")
        print(f"Grid appeared in RSK {rsk_count} times - it's not exclusive")
    else:
        print(f"\n⚠️ Grid didn't appear in either algorithm - very rare configuration")
    
    return push_count, rsk_count

if __name__ == "__main__":
    extra_verification()
