from schur_sampler import sample_push_block_grid, sample_rsk_grid
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter, defaultdict
import argparse
import sys

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

def compare_full_grid_distributions(X, Y, sim_count=1000, debug=False, ignore_edges=True):
    """
    Compare the full grid distributions of push-block and RSK algorithms using complete partitions.
    
    Parameters:
    - X: List of probabilities for each row in the grid
    - Y: List of probabilities for each column in the grid
    - sim_count: Number of simulations to run
    - debug: Boolean flag to enable debug mode
    - ignore_edges: Boolean flag to ignore edge partitions (first/last rows/columns)
    
    Returns:
    - Dictionary with comparison statistics
    """
    rows = len(Y) + 1
    cols = len(X) + 1
    
    # Store the complete partitions for each grid position
    push_block_partitions = [[[] for _ in range(cols)] for _ in range(rows)]
    rsk_partitions = [[[] for _ in range(cols)] for _ in range(rows)]
    
    print(f"Running {sim_count} simulations on {rows}x{cols} grid (complete partitions)...")
    if ignore_edges:
        print("Ignoring edge partitions in analysis.")
    
    for sim in range(sim_count):
        if sim % (sim_count // 10) == 0 and sim > 0:
            print(f"Progress: {sim}/{sim_count} ({100*sim//sim_count}%)")
        
        # Sample both grids
        push_block_grid = sample_push_block_grid(X, Y)
        rsk_grid = sample_rsk_grid(X, Y)
        
        # Extract complete partitions for each position
        for i in range(rows):
            for j in range(cols):
                push_partition = tuple(push_block_grid[i][j]._parts)
                rsk_partition = tuple(rsk_grid[i][j]._parts)
                push_block_partitions[i][j].append(push_partition)
                rsk_partitions[i][j].append(rsk_partition)
    
    # Compute statistics for each position
    stats = {}
    total_kl_divergence = 0
    total_jaccard_similarity = 0
    max_kl_divergence = 0
    min_jaccard_similarity = 1
    max_kl_pos = None
    min_jaccard_pos = None
    analyzed_positions = 0
    
    for i in range(rows):
        for j in range(cols):
            # Skip edge positions if ignore_edges is True
            if ignore_edges and (i == 0 or i == rows-1 or j == 0 or j == cols-1):
                continue
                
            push_counts = Counter(push_block_partitions[i][j])
            rsk_counts = Counter(rsk_partitions[i][j])
            
            push_unique = set(push_block_partitions[i][j])
            rsk_unique = set(rsk_partitions[i][j])
            common_partitions = push_unique & rsk_unique
            all_partitions = push_unique | rsk_unique
            
            # Calculate metrics
            jaccard_similarity = len(common_partitions) / len(all_partitions) if all_partitions else 1
            
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
                        return float('inf')
                return kl
            
            kl_divergence = calculate_kl_divergence(push_counts, rsk_counts, all_partitions)
            
            # Update totals (ignore infinite KL divergences for averages)
            if kl_divergence != float('inf'):
                total_kl_divergence += kl_divergence
                if kl_divergence > max_kl_divergence:
                    max_kl_divergence = kl_divergence
                    max_kl_pos = (i, j)
            
            total_jaccard_similarity += jaccard_similarity
            if jaccard_similarity < min_jaccard_similarity:
                min_jaccard_similarity = jaccard_similarity
                min_jaccard_pos = (i, j)
            
            analyzed_positions += 1
            
            stats[(i, j)] = {
                'push_block_unique': len(push_unique),
                'rsk_unique': len(rsk_unique),
                'common_partitions': len(common_partitions),
                'jaccard_similarity': jaccard_similarity,
                'kl_divergence': kl_divergence,
                'push_block_counts': push_counts,
                'rsk_counts': rsk_counts,
                'push_block_data': push_block_partitions[i][j],
                'rsk_data': rsk_partitions[i][j]
            }
    
    summary = {
        'total_kl_divergence': total_kl_divergence,
        'total_jaccard_similarity': total_jaccard_similarity,
        'avg_kl_divergence': total_kl_divergence / analyzed_positions if analyzed_positions > 0 else 0,
        'avg_jaccard_similarity': total_jaccard_similarity / analyzed_positions if analyzed_positions > 0 else 0,
        'max_kl_divergence': max_kl_divergence,
        'min_jaccard_similarity': min_jaccard_similarity,
        'max_kl_pos': max_kl_pos,
        'min_jaccard_pos': min_jaccard_pos,
        'grid_size': (rows, cols),
        'analyzed_positions': analyzed_positions,
        'ignore_edges': ignore_edges,
        'stats': stats
    }
    
    return summary

def plot_grid_comparison(stats_summary, X, Y, save_path=None):
    """
    Plot heatmaps comparing the two algorithms across the full grid using partition metrics.
    
    Parameters:
    - stats_summary: Dictionary returned by compare_full_grid_distributions
    - X, Y: Grid parameters
    - save_path: Optional path to save the plot
    """
    rows, cols = stats_summary['grid_size']
    stats = stats_summary['stats']
    ignore_edges = stats_summary.get('ignore_edges', False)
    
    # Create matrices for the heatmaps
    push_block_unique = np.full((rows, cols), np.nan)
    rsk_unique = np.full((rows, cols), np.nan)
    jaccard_similarities = np.full((rows, cols), np.nan)
    kl_divergences = np.full((rows, cols), np.nan)
    
    for (i, j), data in stats.items():
        push_block_unique[i, j] = data['push_block_unique']
        rsk_unique[i, j] = data['rsk_unique']
        jaccard_similarities[i, j] = data['jaccard_similarity']
        kl_div = data['kl_divergence']
        kl_divergences[i, j] = kl_div if kl_div != float('inf') else np.nan
    
    # Create the plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Push-block unique partitions heatmap
    im1 = axes[0, 0].imshow(push_block_unique, cmap='viridis', aspect='auto')
    axes[0, 0].set_title('Push-Block Algorithm: Unique Partitions Count')
    axes[0, 0].set_xlabel('Column Index (m)')
    axes[0, 0].set_ylabel('Row Index (n)')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # RSK unique partitions heatmap
    im2 = axes[0, 1].imshow(rsk_unique, cmap='viridis', aspect='auto')
    axes[0, 1].set_title('RSK Algorithm: Unique Partitions Count')
    axes[0, 1].set_xlabel('Column Index (m)')
    axes[0, 1].set_ylabel('Row Index (n)')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Jaccard similarity heatmap
    im3 = axes[1, 0].imshow(jaccard_similarities, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    axes[1, 0].set_title('Jaccard Similarity (1 = identical distributions)')
    axes[1, 0].set_xlabel('Column Index (m)')
    axes[1, 0].set_ylabel('Row Index (n)')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # KL divergence heatmap
    with np.errstate(invalid='ignore'):
        im4 = axes[1, 1].imshow(kl_divergences, cmap='Reds', aspect='auto')
    axes[1, 1].set_title('KL Divergence (0 = identical distributions)')
    axes[1, 1].set_xlabel('Column Index (m)')
    axes[1, 1].set_ylabel('Row Index (n)')
    plt.colorbar(im4, ax=axes[1, 1])
    
    plt.tight_layout()
    edge_text = " (excluding edges)" if ignore_edges else ""
    plt.suptitle(f'Grid Comparison (Complete Partitions): X={X}, Y={Y}{edge_text}', fontsize=16, y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()
    
    plt.show()

def print_grid_summary(stats_summary, X, Y):
    """
    Print a summary of the grid comparison results for both algorithms using complete partitions.
    """
    print(f"\nGrid Comparison Summary (Complete Partitions)")
    print(f"=" * 70)
    print(f"Grid parameters: X={X}, Y={Y}")
    print(f"Grid size: {stats_summary['grid_size'][0]} x {stats_summary['grid_size'][1]}")
    print(f"Total KL divergence: {stats_summary['total_kl_divergence']:.4f}")
    print(f"Total Jaccard similarity: {stats_summary['total_jaccard_similarity']:.4f}")
    print(f"Average KL divergence: {stats_summary['avg_kl_divergence']:.4f}")
    print(f"Average Jaccard similarity: {stats_summary['avg_jaccard_similarity']:.4f}")
    print(f"Maximum KL divergence: {stats_summary['max_kl_divergence']:.4f} at {stats_summary['max_kl_pos']}")
    print(f"Minimum Jaccard similarity: {stats_summary['min_jaccard_similarity']:.4f} at {stats_summary['min_jaccard_pos']}")
    
    # Show top 5 positions with largest differences
    stats = stats_summary['stats']
    
    # Sort by KL divergence (excluding infinite values)
    finite_kl_stats = [(pos, data) for pos, data in stats.items() 
                       if data['kl_divergence'] != float('inf')]
    sorted_kl = sorted(finite_kl_stats, key=lambda x: x[1]['kl_divergence'], reverse=True)
    
    # Sort by Jaccard similarity (ascending - less similar first)
    sorted_jaccard = sorted(stats.items(), key=lambda x: x[1]['jaccard_similarity'])
    
    print(f"\nTop 5 positions with largest KL divergences:")
    print(f"{'Position':<12} {'Push Unique':<12} {'RSK Unique':<12} {'KL Divergence':<15}")
    print(f"{'-'*55}")
    
    for i, ((row, col), data) in enumerate(sorted_kl[:5]):
        print(f"({row:2d},{col:2d})      {data['push_block_unique']:8d}     "
              f"{data['rsk_unique']:8d}     {data['kl_divergence']:8.4f}")
    
    print(f"\nTop 5 positions with lowest Jaccard similarities:")
    print(f"{'Position':<12} {'Push Unique':<12} {'RSK Unique':<12} {'Jaccard Sim.':<15}")
    print(f"{'-'*55}")
    
    for i, ((row, col), data) in enumerate(sorted_jaccard[:5]):
        print(f"({row:2d},{col:2d})      {data['push_block_unique']:8d}     "
              f"{data['rsk_unique']:8d}     {data['jaccard_similarity']:8.4f}")

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
    grid_parser = subparsers.add_parser('grid', help='Compare algorithms across the full grid')
    grid_parser.add_argument('--X', type=float, nargs='+', default=[0.5, 0.5],
                           help='X parameters (space-separated). Default: 0.5 0.5')
    grid_parser.add_argument('--Y', type=float, nargs='+', default=[0.5, 0.5],
                           help='Y parameters (space-separated). Default: 0.5 0.5')
    grid_parser.add_argument('--simulations', type=int, default=1000,
                           help='Number of simulations. Default: 1000')
    grid_parser.add_argument('--plot', action='store_true',
                           help='Show grid comparison heatmaps')
    grid_parser.add_argument('--save', type=str,
                           help='Save plot to specified path')
    grid_parser.add_argument('--ignore-edges', action='store_true',
                           help='Ignore edge partitions (first/last rows/columns) in analysis')
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
    for i in range(sim_count):
        if i % (sim_count // 10) == 0 and i > 0:
            print(f"Progress: {i}/{sim_count} ({100*i//sim_count}%)")
            
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
    print_partition_summary(push_block_partitions[:min_len], rsk_partitions[:min_len], path)

def grid_based_comparison(args):
    """Run the new grid-based comparison."""
    X, Y = args.X, args.Y
    sim_count = args.simulations
    debug = args.debug
    plot = args.plot
    save_path = args.save
    ignore_edges = getattr(args, 'ignore_edges', False)  # Handle the hyphenated argument
    
    print(f"Running grid-based comparison:")
    print(f"X={X}, Y={Y}, Simulations={sim_count}")
    if ignore_edges:
        print("Ignoring edge partitions in analysis")
    
    # Run the comparison
    stats_summary = compare_full_grid_distributions(X, Y, sim_count, debug, ignore_edges)
    
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
    print(f"\nPartition Distribution Summary for Path: {path}")
    print("="*70)
    
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