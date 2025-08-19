#!/usr/bin/env python3
"""
Test different parameter sets to find the most distinctive exclusive grids.
"""

from show_grid_details import find_exclusive_grids, analyze_exclusive_grids

def test_parameter_sets():
    """Test different parameter combinations to find the best examples of exclusive grids."""
    
    parameter_sets = [
        ([0.3, 0.4], [0.3, 0.4], "Balanced small"),
        ([0.2, 0.6], [0.2, 0.6], "Asymmetric"),
        ([0.1, 0.2], [0.7, 0.8], "Very asymmetric"),
        ([0.4, 0.5], [0.1, 0.1], "Small Y values"),
        ([0.1, 0.1], [0.4, 0.5], "Small X values"),
    ]
    
    best_results = None
    best_exclusivity = 0
    
    for X, Y, description in parameter_sets:
        print(f"\n{'='*80}")
        print(f"TESTING: {description} - X={X}, Y={Y}")
        print(f"{'='*80}")
        
        try:
            push_only, rsk_only, common = find_exclusive_grids(X, Y, num_samples=2000)
            
            total_exclusive = len(push_only) + len(rsk_only)
            total_grids = len(push_only) + len(rsk_only) + len(common)
            exclusivity_ratio = total_exclusive / total_grids if total_grids > 0 else 0
            
            print(f"Exclusivity ratio: {exclusivity_ratio:.3f} ({total_exclusive}/{total_grids})")
            
            if exclusivity_ratio > best_exclusivity:
                best_exclusivity = exclusivity_ratio
                best_results = (X, Y, description, push_only, rsk_only, common)
            
            # Show a few examples if there are exclusive grids
            if push_only or rsk_only:
                analyze_exclusive_grids(push_only, rsk_only, max_display=2)
            else:
                print("❌ No exclusive grids found for this parameter set")
                
        except Exception as e:
            print(f"❌ Error with parameters X={X}, Y={Y}: {e}")
    
    if best_results:
        X, Y, description, push_only, rsk_only, common = best_results
        print(f"\n{'='*80}")
        print(f"BEST EXCLUSIVITY: {description} - X={X}, Y={Y}")
        print(f"Exclusivity ratio: {best_exclusivity:.3f}")
        print(f"{'='*80}")
        analyze_exclusive_grids(push_only, rsk_only, max_display=5)

if __name__ == "__main__":
    test_parameter_sets()
