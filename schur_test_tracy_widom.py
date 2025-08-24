from schur_sampler import sample_rsk_grid
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter, defaultdict
import argparse
import sys
import math
import time
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
from scipy.stats import gaussian_kde

# get the maximum sum of all geometrics paths


def run_single_simulation(args_tuple):
    """Run a single Tracy-Widom simulation using RSK only (multiprocessing-friendly)."""
    M, N, x_i, y_i, shared_q_grid, shared_X, shared_Y = args_tuple

    # Use shared X and Y arrays for RSK if provided, otherwise use fixed x_i,y_i
    if shared_X is not None and shared_Y is not None:
        X = shared_X
        Y = shared_Y
    else:
        X = [x_i] * M  # Convert to list, not numpy array
        Y = [y_i] * N  # Convert to list, not numpy array

    try:
        grid = sample_rsk_grid(X, Y)
        # Fix: grid is (N+1) x (M+1), so the bottom-right corner is grid[N][M]
        bottom_right_partition = grid[N][M]
        lambda_1 = bottom_right_partition.part(1) if hasattr(bottom_right_partition, 'part') else 0

        # Fix: q should be a scalar value, not an array
        q = x_i if shared_X is None else np.mean(shared_X)
        
        # Ensure all calculations return scalars
        sub = (2 * q * N) / (1 - q) 
        sigma_q = (pow(q, 1/3) * pow(1 + q, 1/3)) / (1 - q)
        f_q = pow(q, 1/3) / (2 * pow(1 + q, 2/3))

        p = q / (1 - q)
        sigma = pow(p * (1 + p), 1/2)

        res = (lambda_1 - sub) * pow(N, -1/3) / sigma / pow(f_q, 1/2)

        # Ensure we return a scalar float, not a numpy array
        # return float(res)
        return float(res)
    except Exception as e:
        print(f"RSK error: {e}")
        return 0.0

# Geometric-sampling helpers removed: script now uses RSK sampler only.

def main():
    parser = argparse.ArgumentParser(description="Tracy-Widom test for maximum path sums in geometric grids")
    parser.add_argument('--M', type=int, default=5, help='Grid width (default: 5)')
    parser.add_argument('--N', type=int, default=5, help='Grid height (default: 5)')
    parser.add_argument('--x', type=float, default=0.5, help='x_i parameter (default: 0.5)')
    parser.add_argument('--y', type=float, default=0.5, help='y_i parameter (default: 0.5)')
    parser.add_argument('--random-q', action='store_true', help='Use random q ~ Uniform(0,1) for each simulation instead of fixed q=x*y')
    parser.add_argument('--simulations', type=int, default=100, help='Number of simulations (default: 100)')
    parser.add_argument('--plot', action='store_true', help='Show histogram plot')
    parser.add_argument('--save-plot', type=str, help='Save plot to file (e.g., histogram.png)')
    parser.add_argument('--bins', type=int, default=None, help='Number of histogram bins (default: auto)')
    parser.add_argument('--processes', type=int, default=None, help='Number of processes to use (default: CPU count)')
    parser.add_argument('--chunk-size', type=int, default=None, help='Chunk size for multiprocessing (default: auto)')
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmark')
    
    args = parser.parse_args()
    
    # Run benchmark if requested
    if args.benchmark:
        benchmark_performance(args.M, args.N, args.x, args.y, min(args.simulations, 200))
        return
    
    M, N = args.M, args.N
    x_i, y_i = args.x, args.y
    sim_count = args.simulations
    num_processes = args.processes or mp.cpu_count()
    use_random_q = args.random_q
    chunk_size = args.chunk_size or max(1, sim_count // (num_processes * 4))
    
    max_sums = []
    
    print(f"Running Tracy-Widom test with {M}x{N} grid")
    if use_random_q:
        print(f"Parameters: Using shared random q grid ~ Uniform(0,1)")
    else:
        print(f"Parameters: x_i={x_i}, y_i={y_i}, q={x_i*y_i}")
    print(f"Simulations: {sim_count}")
    print(f"Processes: {num_processes}")
    print(f"Chunk size: {chunk_size}")
    print("Method: RSK sampler (only method supported)")
    
    start_time = time.perf_counter()
    
    # Generate shared q grid if using random q
    shared_q_grid = None
    shared_X = None
    shared_Y = None
    if use_random_q:
        print("Generating shared random parameters for RSK sampler...")
        # For RSK sampler, create 1D lists (not numpy arrays) of random values
        shared_X = [np.random.uniform(0, 1) for _ in range(M)]
        shared_Y = [np.random.uniform(0, 1) for _ in range(N)]
        print(f"Generated X array (length {M}) with values in range [{min(shared_X):.3f}, {max(shared_X):.3f}]")
        print(f"Generated Y array (length {N}) with values in range [{min(shared_Y):.3f}, {max(shared_Y):.3f}]")
    
    # Smart parallel processing decision
    if num_processes == 1 or sim_count < 50:
        # Use sequential for small jobs or when explicitly requested
        use_parallel = False
    else:
        use_parallel = True
    
    # Prepare arguments for parallel processing (RSK-only signature)
    sim_args = [(M, N, x_i, y_i, shared_q_grid, shared_X, shared_Y) for _ in range(sim_count)]

    # Run simulations
    if use_parallel:
        print(f"Running parallel simulations...")
        with mp.Pool(processes=num_processes) as pool:
            # Use imap for progress tracking
            max_sums = list(tqdm(
                pool.imap(run_single_simulation, sim_args, chunksize=chunk_size),
                total=sim_count,
                desc="Tracy-Widom Parallel",
                unit="sim"
            ))
    else:
        print(f"Running sequential simulations...")
        max_sums = [run_single_simulation(args) for args in tqdm(sim_args, desc="Tracy-Widom Sequential", unit="sim")]
    
    end_time = time.perf_counter()
    
    print(f"Simulation completed in {end_time - start_time:.2f} seconds")
    print(f"Average time per simulation: {(end_time - start_time) / sim_count * 1000:.2f} ms")

    # Basic statistics
    print(f"\nResults:")
    print(f"Mean maximum sum: {np.mean(max_sums):.4f}")
    print(f"Std maximum sum: {np.std(max_sums):.4f}")
    print(f"Min maximum sum: {np.min(max_sums):.4f}")
    print(f"Max maximum sum: {np.max(max_sums):.4f}")
    
    # Plot histogram
    if args.plot or args.save_plot:
        # Convert to numpy array and ensure all values are scalars
        max_sums_array = np.array([float(x) for x in max_sums])
        
        plt.figure(figsize=(12, 8))
        
        # Create both histogram and smoothed density plot
        # First, create histogram for reference
        if args.bins:
            bins = args.bins
            print(f"Using manually specified {bins} bins")
        else:
            unique_values = len(set(max_sums_array))  # Now using the cleaned array
            data_range = np.max(max_sums_array) - np.min(max_sums_array)
            
            # Choose number of bins intelligently
            if unique_values <= 20:
                # Few unique values - use one bin per value
                bins = unique_values
            elif data_range <= 50:
                # Small range - use fine granularity
                bins = min(50, int(data_range) + 1)
            else:
                # Large range - use Sturges' rule or square root rule
                bins = min(unique_values, 100)  # Cap at 100 bins max
            
            print(f"Auto-selected {bins} bins for {unique_values} unique values (range: {np.min(max_sums_array):.1f} - {np.max(max_sums_array):.1f})")
        
        # Create histogram with lower alpha for background
        n, bins_edges, patches = plt.hist(max_sums_array, bins=bins, alpha=0.3, density=True, 
                                         edgecolor='black', linewidth=0.5, color='lightblue', 
                                         label='Histogram')
        
        # Create smoothed density plot using KDE (only if data has variance)
        if np.std(max_sums_array) > 1e-10:  # Check if data has meaningful variance
            kde = gaussian_kde(max_sums_array)
            
            # Create dense x-axis for smooth curve covering the full range
            x_min, x_max = np.min(max_sums_array), np.max(max_sums_array)
            x_range = x_max - x_min
            if x_range > 1e-10:  # Only create smooth curve if there's a meaningful range
                x_smooth = np.linspace(x_min - 0.1 * x_range, x_max + 0.1 * x_range, 1000)
                
                # Calculate density values
                density_smooth = kde(x_smooth)
                
                # Plot smooth density curve
                plt.plot(x_smooth, density_smooth, 'b-', linewidth=2, label='Smoothed Density (KDE)')
                plt.fill_between(x_smooth, density_smooth, alpha=0.2, color='blue')
            else:
                print("Warning: Data has no range - skipping KDE curve")
        else:
            print("Warning: Data has no variance - skipping KDE curve")
        
        # Add statistics overlay - use the cleaned array
        mean_val = np.mean(max_sums_array)
        std_val = np.std(max_sums_array)
        plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_val:.2f}')
        plt.axvline(mean_val + std_val, color='orange', linestyle='--', alpha=0.7,
                   label=f'Mean + σ: {mean_val + std_val:.2f}')
        plt.axvline(mean_val - std_val, color='orange', linestyle='--', alpha=0.7,
                   label=f'Mean - σ: {mean_val - std_val:.2f}')
        
        title_suffix = "Shared q grid ~ U(0,1)" if use_random_q else f"q={x_i*y_i}"
        plt.title(f'Distribution of Maximum λ₁ ({M}×{N} grid, {sim_count} simulations)\n{title_suffix}, Method: RSK')
        plt.xlabel('Maximum λ₁')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add text box with statistics
        stats_text = f'Samples: {len(max_sums_array)}\nMean: {mean_val:.3f}\nStd: {std_val:.3f}\nMin: {np.min(max_sums_array):.1f}\nMax: {np.max(max_sums_array):.1f}'
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                verticalalignment='top', fontsize=10)
        
        plt.tight_layout()
        
        if args.save_plot:
            plt.savefig(args.save_plot, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {args.save_plot}")
        
        if args.plot:
            plt.show()
        else:
            plt.close()  # Close figure if only saving
    
    return max_sums

# All geometric helper functions removed; script simplified to RSK-only flow.

def benchmark_performance(M, N, x_i, y_i, sim_count=100):
    """Benchmark sequential vs parallel performance."""
    print(f"\nBenchmarking performance for {M}x{N} grid with {sim_count} simulations")
    
    # Test sequential
    start_time = time.perf_counter()
    sim_args = [(M, N, x_i, y_i, None, None, None) for _ in range(sim_count)]
    sequential_results = [run_single_simulation(args) for args in sim_args]
    sequential_time = time.perf_counter() - start_time
    
    # Test parallel
    num_processes = mp.cpu_count()
    start_time = time.perf_counter()
    with mp.Pool(processes=num_processes) as pool:
        parallel_results = pool.map(run_single_simulation, sim_args)
    parallel_time = time.perf_counter() - start_time
    
    speedup = sequential_time / parallel_time
    print(f"Sequential time: {sequential_time:.2f}s")
    print(f"Parallel time ({num_processes} cores): {parallel_time:.2f}s")
    print(f"Speedup: {speedup:.2f}x")
    print(f"Efficiency: {speedup/num_processes*100:.1f}%")
    
    return speedup


if __name__ == "__main__":
    main()