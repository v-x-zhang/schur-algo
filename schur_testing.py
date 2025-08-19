from schur_sampler import sample_push_block_grid, sample_rsk_grid
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter, defaultdict
import argparse
import sys
from tqdm import tqdm

def sample_both_grids(X, Y, path, debug=False):
    """
    Sample push block and RSK grids based on the provided path.
    
    Parameters:
    - X: List of probabilities for each row in the grid.
    - Y: List of probabilities for each column in the grid.
    - path: The path to follow in the grid (a string of 'D' and 'R' characters).
    - debug: Boolean flag to enable debug mode.
    Returns:
    - the push block path partitions and RSK path partitions.
    """

    push_block_grid = sample_push_block_grid(X, Y)
    rows = len(push_block_grid)
    cols = len(push_block_grid[0])
    
    # Start at the top-left corner of the grid (final row, first column)
    current_spot = (rows - 1, 0)
    push_block_path_partitions = []
    
    # Add the starting partition
    push_block_path_partitions.append((current_spot, push_block_grid[current_spot[0]][current_spot[1]]))
    if debug:
        print(f"Push-Block Start: {current_spot} -> {push_block_grid[current_spot[0]][current_spot[1]]}")

    for step in path:
        if step == 'D':
            # Move down in the grid (decrease row index)
            current_spot = (current_spot[0] - 1, current_spot[1])
        elif step == 'R':
            # Move right in the grid (increase column index)
            current_spot = (current_spot[0], current_spot[1] + 1)
        
        # Check bounds
        if 0 <= current_spot[0] < rows and 0 <= current_spot[1] < cols:
            # Append the current partition to the path partitions
            push_block_path_partitions.append((current_spot, push_block_grid[current_spot[0]][current_spot[1]]))
            if debug:
                print(f"Push-Block Step {step}: {current_spot} -> {push_block_grid[current_spot[0]][current_spot[1]]}")
        else:
            if debug:
                print(f"Push-Block Step {step}: {current_spot} -> Out of bounds!")
            break

    # Sample an RSK grid
    rsk_grid = sample_rsk_grid(X, Y)
    current_spot = (rows - 1, 0)  # Start at the top-left corner of the grid
    rsk_path_partitions = []
    
    # Add the starting partition
    rsk_path_partitions.append((current_spot, rsk_grid[current_spot[0]][current_spot[1]]))
    if debug:
        print(f"RSK Start: {current_spot} -> {rsk_grid[current_spot[0]][current_spot[1]]}")

    for step in path:
        if step == 'D':
            # Move down in the grid (decrease row index)
            current_spot = (current_spot[0] - 1, current_spot[1])
        elif step == 'R':
            # Move right in the grid (increase column index)
            current_spot = (current_spot[0], current_spot[1] + 1)
        
        # Check bounds
        if 0 <= current_spot[0] < rows and 0 <= current_spot[1] < cols:
            # Append the current partition to the path partitions
            rsk_path_partitions.append((current_spot, rsk_grid[current_spot[0]][current_spot[1]]))
            if debug:
                print(f"RSK Step {step}: {current_spot} -> {rsk_grid[current_spot[0]][current_spot[1]]}")
        else:
            if debug:
                print(f"RSK Step {step}: {current_spot} -> Out of bounds!")
            break

    return push_block_path_partitions, rsk_path_partitions

def compare_grid_distributions(X, Y, sim_count=1000, debug=False):
    """
    Compare the complete grid distributions of push-block and RSK algorithms.
    
    Parameters:
    - X: List of probabilities for each row in the grid
    - Y: List of probabilities for each column in the grid
    - sim_count: Number of simulations to run
    - debug: Boolean flag to enable debug mode
    
    Returns:
    - Dictionary with comparison statistics
    """
    rows = len(Y) + 1
    cols = len(X) + 1
    
    # Store the complete grids as hashable tuples
    push_block_grids = []
    rsk_grids = []
    
    print(f"Running {sim_count} simulations on {rows}x{cols} grid (complete grid distributions)...")
    
    # Use tqdm for progress monitoring with time estimates
    for sim in tqdm(range(sim_count), desc="Grid Simulation", unit="sim"):
        # Sample both grids
        push_block_grid = sample_push_block_grid(X, Y)
        rsk_grid = sample_rsk_grid(X, Y)
        
        # Convert grids to hashable format (tuple of tuples of partition tuples)
        push_grid_tuple = tuple(
            tuple(tuple(partition._parts) for partition in row) 
            for row in push_block_grid
        )
        rsk_grid_tuple = tuple(
            tuple(tuple(partition._parts) for partition in row) 
            for row in rsk_grid
        )
        
        push_block_grids.append(push_grid_tuple)
        rsk_grids.append(rsk_grid_tuple)
    
    # Count unique grid configurations
    push_counts = Counter(push_block_grids)
    rsk_counts = Counter(rsk_grids)
    
    push_unique = set(push_block_grids)
    rsk_unique = set(rsk_grids)
    common_grids = push_unique & rsk_unique
    all_grids = push_unique | rsk_unique
    
    # Calculate metrics
    jaccard_similarity = len(common_grids) / len(all_grids) if all_grids else 1
    
    # Calculate KL divergence between grid distributions
    def calculate_kl_divergence(p_dist, q_dist, all_items):
        kl = 0
        total_p = sum(p_dist.values())
        total_q = sum(q_dist.values())
        
        for item in all_items:
            p_prob = p_dist.get(item, 0) / total_p if total_p > 0 else 0
            q_prob = q_dist.get(item, 0) / total_q if total_q > 0 else 0
            
            if p_prob > 0 and q_prob > 0:
                kl += p_prob * np.log2(p_prob / q_prob)
            elif p_prob > 0 and q_prob == 0:
                return float('inf')
        return kl
    
    kl_divergence = calculate_kl_divergence(push_counts, rsk_counts, all_grids)
    
    # Calculate entropy for each distribution
    def calculate_entropy(counts, total):
        if total == 0:
            return 0
        entropy = 0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p)
        return entropy
    
    push_entropy = calculate_entropy(push_counts, len(push_block_grids))
    rsk_entropy = calculate_entropy(rsk_counts, len(rsk_grids))
    
    summary = {
        'push_block_unique': len(push_unique),
        'rsk_unique': len(rsk_unique),
        'common_grids': len(common_grids),
        'total_grids': len(all_grids),
        'jaccard_similarity': jaccard_similarity,
        'kl_divergence': kl_divergence,
        'push_entropy': push_entropy,
        'rsk_entropy': rsk_entropy,
        'grid_size': (rows, cols),
        'simulations': sim_count,
        'push_counts': push_counts,
        'rsk_counts': rsk_counts,
        'push_grids': push_block_grids,
        'rsk_grids': rsk_grids
    }
    
    return summary

def plot_grid_comparison(stats_summary, X, Y, save_path=None):
    """
    Plot comparison of complete grid distributions between the two algorithms.
    
    Parameters:
    - stats_summary: Dictionary returned by compare_grid_distributions
    - X, Y: Grid parameters
    - save_path: Optional path to save the plot
    """
    push_counts = stats_summary['push_counts']
    rsk_counts = stats_summary['rsk_counts']
    
    # Get the most common grids for visualization
    max_grids_to_show = 10
    top_push_grids = push_counts.most_common(max_grids_to_show)
    top_rsk_grids = rsk_counts.most_common(max_grids_to_show)
    
    # Create the plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Distribution of unique grid counts
    unique_counts = [stats_summary['push_block_unique'], stats_summary['rsk_unique'], 
                    stats_summary['common_grids']]
    labels = ['Push-Block\nUnique', 'RSK\nUnique', 'Common\nGrids']
    colors = ['blue', 'red', 'green']
    
    axes[0, 0].bar(labels, unique_counts, color=colors, alpha=0.7)
    axes[0, 0].set_title('Grid Distribution Comparison')
    axes[0, 0].set_ylabel('Number of Unique Grids')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add values on top of bars
    for i, v in enumerate(unique_counts):
        axes[0, 0].text(i, v + max(unique_counts) * 0.01, str(v), 
                       ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Entropy comparison
    entropies = [stats_summary['push_entropy'], stats_summary['rsk_entropy']]
    entropy_labels = ['Push-Block', 'RSK']
    
    axes[0, 1].bar(entropy_labels, entropies, color=['blue', 'red'], alpha=0.7)
    axes[0, 1].set_title('Grid Distribution Entropy')
    axes[0, 1].set_ylabel('Entropy (bits)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Add values on top of bars
    for i, v in enumerate(entropies):
        axes[0, 1].text(i, v + max(entropies) * 0.01, f'{v:.3f}', 
                       ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Top Push-Block grid frequencies
    if top_push_grids:
        push_probs = [count / stats_summary['simulations'] for grid, count in top_push_grids]
        grid_indices = range(len(top_push_grids))
        
        axes[1, 0].bar(grid_indices, push_probs, color='blue', alpha=0.7)
        axes[1, 0].set_title(f'Top {len(top_push_grids)} Push-Block Grid Configurations')
        axes[1, 0].set_xlabel('Grid Configuration (ranked by frequency)')
        axes[1, 0].set_ylabel('Probability')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add probability labels
        for i, prob in enumerate(push_probs):
            axes[1, 0].text(i, prob + max(push_probs) * 0.01, f'{prob:.3f}', 
                           ha='center', va='bottom', fontsize=8)
    
    # Plot 4: Top RSK grid frequencies
    if top_rsk_grids:
        rsk_probs = [count / stats_summary['simulations'] for grid, count in top_rsk_grids]
        grid_indices = range(len(top_rsk_grids))
        
        axes[1, 1].bar(grid_indices, rsk_probs, color='red', alpha=0.7)
        axes[1, 1].set_title(f'Top {len(top_rsk_grids)} RSK Grid Configurations')
        axes[1, 1].set_xlabel('Grid Configuration (ranked by frequency)')
        axes[1, 1].set_ylabel('Probability')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add probability labels
        for i, prob in enumerate(rsk_probs):
            axes[1, 1].text(i, prob + max(rsk_probs) * 0.01, f'{prob:.3f}', 
                           ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.suptitle(f'Complete Grid Distribution Comparison: X={X}, Y={Y}', 
                 fontsize=16, y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()

def print_grid_summary(stats_summary, X, Y):
    """
    Print a summary of the complete grid distribution comparison results.
    """
    print(f"\nComplete Grid Distribution Comparison Summary")
    print(f"=" * 70)
    print(f"Grid parameters: X={X}, Y={Y}")
    print(f"Grid size: {stats_summary['grid_size'][0]} x {stats_summary['grid_size'][1]}")
    print(f"Number of simulations: {stats_summary['simulations']}")
    print(f"Push-Block unique grids: {stats_summary['push_block_unique']}")
    print(f"RSK unique grids: {stats_summary['rsk_unique']}")
    print(f"Common grids: {stats_summary['common_grids']}")
    print(f"Total unique grids: {stats_summary['total_grids']}")
    print(f"Jaccard similarity: {stats_summary['jaccard_similarity']:.4f}")
    print(f"KL divergence: {stats_summary['kl_divergence']:.4f}" if stats_summary['kl_divergence'] != float('inf') else "KL divergence: ∞")
    print(f"Push-Block entropy: {stats_summary['push_entropy']:.4f}")
    print(f"RSK entropy: {stats_summary['rsk_entropy']:.4f}")
    
    # Show top grid configurations
    push_counts = stats_summary['push_counts']
    rsk_counts = stats_summary['rsk_counts']
    
    print(f"\nTop 5 Push-Block grid configurations:")
    print(f"{'Rank':<6} {'Frequency':<12} {'Probability':<12} {'Grid Structure'}")
    print(f"{'-'*80}")
    
    for i, (grid, count) in enumerate(push_counts.most_common(5)):
        prob = count / stats_summary['simulations']
        grid_str = f"{len(grid)}x{len(grid[0])} grid" if grid else "Empty grid"
        print(f"{i+1:<6} {count:<12} {prob:<12.4f} {grid_str}")
    
    print(f"\nTop 5 RSK grid configurations:")
    print(f"{'Rank':<6} {'Frequency':<12} {'Probability':<12} {'Grid Structure'}")
    print(f"{'-'*80}")
    
    for i, (grid, count) in enumerate(rsk_counts.most_common(5)):
        prob = count / stats_summary['simulations']
        grid_str = f"{len(grid)}x{len(grid[0])} grid" if grid else "Empty grid"
        print(f"{i+1:<6} {count:<12} {prob:<12.4f} {grid_str}")
    
    # Check if the most common grids are the same
    top_push = push_counts.most_common(1)[0][0] if push_counts else None
    top_rsk = rsk_counts.most_common(1)[0] if rsk_counts else None
    
    if top_push and top_rsk and top_push == top_rsk[0]:
        print(f"\n✓ Both algorithms have the same most frequent grid configuration!")
    else:
        print(f"\n✗ The algorithms have different most frequent grid configurations.")

def main():
    parser = argparse.ArgumentParser(
        description="Compare Push-Block and RSK algorithms for Schur processes using complete partition analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare complete partitions along a specific path
  python schur_testing.py path --path "DRRD" --X 0.5 0.5 --Y 0.5 0.5 --simulations 1000 --plot

  # Compare full grid partition distributions
  python schur_testing.py grid --X 0.3 0.4 0.2 --Y 0.4 0.5 --simulations 500 --plot

  # Quick test with default parameters
  python schur_testing.py path --simulations 1000
        """
    )
    
    subparsers = parser.add_subparsers(dest='mode', help='Choose comparison mode')
    
    # Path-based comparison (original functionality)
    path_parser = subparsers.add_parser('path', help='Compare algorithms along a specific path')
    path_parser.add_argument('--path', type=str, default="DRRD", 
                           help='Path string (D for down, R for right). Default: "DRRD"')
    path_parser.add_argument('--X', type=float, nargs='+', default=[0.5, 0.5],
                           help='X parameters (space-separated). Default: 0.5 0.5')
    path_parser.add_argument('--Y', type=float, nargs='+', default=[0.5, 0.5],
                           help='Y parameters (space-separated). Default: 0.5 0.5')
    path_parser.add_argument('--simulations', type=int, default=10000,
                           help='Number of simulations. Default: 10000')
    path_parser.add_argument('--plot', action='store_true',
                           help='Show probability distribution plots')
    path_parser.add_argument('--debug', action='store_true',
                           help='Enable debug output')
    
    # Grid-based comparison (new functionality)
    grid_parser = subparsers.add_parser('grid', help='Compare complete grid distributions between algorithms')
    grid_parser.add_argument('--X', type=float, nargs='+', default=[0.3, 0.4],
                           help='X parameters (space-separated). Default: 0.3 0.4 for 3x3 grid')
    grid_parser.add_argument('--Y', type=float, nargs='+', default=[0.3, 0.4],
                           help='Y parameters (space-separated). Default: 0.3 0.4 for 3x3 grid')
    grid_parser.add_argument('--simulations', type=int, default=1000,
                           help='Number of simulations. Default: 1000')
    grid_parser.add_argument('--plot', action='store_true',
                           help='Show grid distribution comparison plots')
    grid_parser.add_argument('--save', type=str,
                           help='Save plot to specified path')
    grid_parser.add_argument('--debug', action='store_true',
                           help='Enable debug output')
    
    args = parser.parse_args()
    
    if args.mode is None:
        parser.print_help()
        return
    
    # Validate X*Y constraints
    for i, x in enumerate(args.X):
        for j, y in enumerate(args.Y):
            if x * y >= 1:
                print(f"Error: Constraint violated: X[{i}] * Y[{j}] = {x * y} >= 1")
                sys.exit(1)
    
    if args.mode == 'path':
        path_based_comparison(args)
    elif args.mode == 'grid':
        grid_based_comparison(args)

def path_based_comparison(args):
    """Run the path-based comparison with both algorithms, analyzing complete partitions."""
    X, Y, path = args.X, args.Y, args.path
    sim_count = args.simulations
    debug = args.debug
    plot = args.plot
    
    print(f"Running path-based comparison (complete partitions):")
    print(f"X={X}, Y={Y}, Path='{path}', Simulations={sim_count}")
    
    # We include the starting position, so we have len(path) + 1 positions
    num_positions = len(path) + 1
    push_block_partitions = [[] for _ in range(num_positions)]
    rsk_partitions = [[] for _ in range(num_positions)]

    # build up distributions of complete partitions for the two samplers
    for i in tqdm(range(sim_count), desc="Path Simulation", unit="sim"):
        push_block_path, rsk_path = sample_both_grids(X, Y, path, debug)
        
        # Make sure we have the expected number of partitions
        min_len = min(len(push_block_path), len(rsk_path), num_positions)
        
        for j in range(min_len):
            # Store the complete partition as a tuple (for hashability)
            push_block_partition = push_block_path[j][1]  # Get the partition object
            rsk_partition = rsk_path[j][1]  # Get the partition object
            
            # Convert partition to tuple for hashing and comparison
            push_block_tuple = tuple(push_block_partition._parts)
            rsk_tuple = tuple(rsk_partition._parts)
            
            push_block_partitions[j].append(push_block_tuple)
            rsk_partitions[j].append(rsk_tuple)

    # Analyze and display the results
    if plot:
        plot_partition_distributions(push_block_partitions[:min_len], rsk_partitions[:min_len], path)

    # Print summary statistics
    if debug:
        print_partition_summary(push_block_partitions[:min_len], rsk_partitions[:min_len], path)

def grid_based_comparison(args):
    """Run the grid-based comparison using complete grid distributions."""
    X, Y = args.X, args.Y
    sim_count = args.simulations
    debug = args.debug
    plot = args.plot
    save_path = args.save
    
    print(f"Running grid-based comparison (complete grid distributions):")
    print(f"X={X}, Y={Y}, Simulations={sim_count}")
    
    # Run the comparison
    stats_summary = compare_grid_distributions(X, Y, sim_count, debug)
    
    # Print summary
    print_grid_summary(stats_summary, X, Y)
    
    # Show plots if requested
    if plot:
        plot_grid_comparison(stats_summary, X, Y, save_path)


def plot_partition_distributions(push_block_data, rsk_data, path):
    """
    Plot probability distributions for complete partitions of both samplers.
    
    Parameters:
    - push_block_data: List of lists containing partition tuples for each step (push-block sampler)
    - rsk_data: List of lists containing partition tuples for each step (RSK sampler)
    - path: The path string used for labeling
    """
    num_steps = len(push_block_data)
    
    # Create subplots
    fig, axes = plt.subplots(2, num_steps, figsize=(6*num_steps, 10))
    if num_steps == 1:
        axes = axes.reshape(2, 1)
    
    for step in range(num_steps):
        # Count frequencies for push-block sampler
        push_counts = Counter(push_block_data[step])
        push_total = len(push_block_data[step])
        
        # Count frequencies for RSK sampler
        rsk_counts = Counter(rsk_data[step])
        rsk_total = len(rsk_data[step])
        
        # Get all unique partitions and sort them
        all_partitions = set(push_block_data[step] + rsk_data[step])
        sorted_partitions = sorted(all_partitions, key=lambda x: (len(x), x) if x else (0,))
        
        # Limit to top partitions to avoid overcrowding
        max_partitions = 15
        if len(sorted_partitions) > max_partitions:
            # Keep the most frequent partitions
            partition_frequencies = {}
            for p in sorted_partitions:
                partition_frequencies[p] = push_counts.get(p, 0) + rsk_counts.get(p, 0)
            sorted_partitions = sorted(sorted_partitions, 
                                     key=lambda x: partition_frequencies[x], 
                                     reverse=True)[:max_partitions]
        
        # Calculate probabilities
        push_probs = [push_counts.get(p, 0) / push_total for p in sorted_partitions]
        rsk_probs = [rsk_counts.get(p, 0) / rsk_total for p in sorted_partitions]
        
        # Create partition labels
        partition_labels = [str(list(p)) if p else "[]" for p in sorted_partitions]
        
        # Create step label
        if step == 0:
            step_label = "Start"
        else:
            step_label = f"{path[step-1]}{step}"
        
        # Plot push-block distribution
        x_pos = np.arange(len(sorted_partitions))
        axes[0, step].bar(x_pos, push_probs, alpha=0.7, color='blue')
        axes[0, step].set_title(f'Push-Block: {step_label}')
        axes[0, step].set_xlabel('Partition')
        axes[0, step].set_ylabel('Probability')
        axes[0, step].set_xticks(x_pos)
        axes[0, step].set_xticklabels(partition_labels, rotation=45, ha='right')
        axes[0, step].grid(True, alpha=0.3)
        axes[0, step].set_ylim(0, 1)
        
        # Plot RSK distribution
        axes[1, step].bar(x_pos, rsk_probs, alpha=0.7, color='red')
        axes[1, step].set_title(f'RSK: {step_label}')
        axes[1, step].set_xlabel('Partition')
        axes[1, step].set_ylabel('Probability')
        axes[1, step].set_xticks(x_pos)
        axes[1, step].set_xticklabels(partition_labels, rotation=45, ha='right')
        axes[1, step].grid(True, alpha=0.3)
        axes[1, step].set_ylim(0, 1)
        
        # Add statistics text
        push_unique = len(push_counts)
        rsk_unique = len(rsk_counts)
        
        axes[0, step].text(0.02, 0.98, f'Unique: {push_unique}', 
                          transform=axes[0, step].transAxes, 
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                          verticalalignment='top')
        axes[1, step].text(0.02, 0.98, f'Unique: {rsk_unique}', 
                          transform=axes[1, step].transAxes,
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                          verticalalignment='top')
    
    plt.tight_layout()
    plt.suptitle(f'Complete Partition Probability Distributions Along Path: {path}', 
                 fontsize=16, y=0.98)
    plt.show()

def print_partition_summary(push_block_data, rsk_data, path):
    """
    Print a summary of the partition distribution comparison.
    """
    # print(f"\nPartition Distribution Summary for Path: {path}")
    # print("="*70)
    
    for step in range(len(push_block_data)):
        # Count partitions
        push_counts = Counter(push_block_data[step])
        rsk_counts = Counter(rsk_data[step])
        
        # Get unique partitions
        push_unique = set(push_block_data[step])
        rsk_unique = set(rsk_data[step])
        common_partitions = push_unique & rsk_unique
        
        # Calculate entropy (diversity measure)
        def calculate_entropy(counts, total):
            if total == 0:
                return 0
            entropy = 0
            for count in counts.values():
                p = count / total
                if p > 0:
                    entropy -= p * np.log2(p)
            return entropy
        
        push_entropy = calculate_entropy(push_counts, len(push_block_data[step]))
        rsk_entropy = calculate_entropy(rsk_counts, len(rsk_data[step]))
        
        # Calculate similarity metrics
        all_partitions = push_unique | rsk_unique
        jaccard_similarity = len(common_partitions) / len(all_partitions) if all_partitions else 0
        
        # Calculate KL divergence
        def calculate_kl_divergence(p_dist, q_dist, all_items):
            kl = 0
            total_p = sum(p_dist.values())
            total_q = sum(q_dist.values())
            
            for item in all_items:
                p_prob = p_dist.get(item, 0) / total_p if total_p > 0 else 0
                q_prob = q_dist.get(item, 0) / total_q if total_q > 0 else 0
                
                if p_prob > 0 and q_prob > 0:
                    kl += p_prob * np.log2(p_prob / q_prob)
                elif p_prob > 0 and q_prob == 0:
                    kl = float('inf')
                    break
            return kl
        
        kl_divergence = calculate_kl_divergence(push_counts, rsk_counts, all_partitions)
        
        if step == 0:
            step_label = "Start"
        else:
            step_label = f"{path[step-1]}{step}"
        
        print(f"{step_label}:")
        print(f"  Push-Block: {len(push_unique)} unique partitions, entropy={push_entropy:.3f}")
        print(f"  RSK:        {len(rsk_unique)} unique partitions, entropy={rsk_entropy:.3f}")
        print(f"  Common partitions: {len(common_partitions)}")
        print(f"  Jaccard similarity: {jaccard_similarity:.3f}")
        print(f"  KL divergence: {kl_divergence:.3f}" if kl_divergence != float('inf') else "  KL divergence: ∞")
        
        # Show most frequent partitions
        print("  Top 5 Push-Block partitions:")
        for partition, count in push_counts.most_common(5):
            prob = count / len(push_block_data[step])
            print(f"    {list(partition) if partition else []}: {prob:.3f}")
        
        print("  Top 5 RSK partitions:")
        for partition, count in rsk_counts.most_common(5):
            prob = count / len(rsk_data[step])
            print(f"    {list(partition) if partition else []}: {prob:.3f}")
        print()


if __name__ == "__main__":
    main()