from schur_sampler import sample_push_block_grid, sample_rsk_grid, sample_push_block_grid_borodin
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import argparse
import sys

def sample_all_grids(X, Y, path, debug):
    """
    Sample push block, RSK, and Borodin grids based on the provided path.
    
    Parameters:
    - X: List of probabilities for each row in the grid.
    - Y: List of probabilities for each column in the grid.
    - path: The path to follow in the grid (a string of 'D' and 'R' characters).
    - debug: Boolean flag to enable debug mode.
    Returns:
    - the push block path partitions, RSK path partitions, and Borodin path partitions.
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

    # Sample a Borodin grid
    borodin_grid = sample_push_block_grid_borodin(X, Y)
    current_spot = (rows - 1, 0)  # Start at the top-left corner of the grid
    borodin_path_partitions = []
    
    # Add the starting partition
    borodin_path_partitions.append((current_spot, borodin_grid[current_spot[0]][current_spot[1]]))
    if debug:
        print(f"Borodin Start: {current_spot} -> {borodin_grid[current_spot[0]][current_spot[1]]}")

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
            borodin_path_partitions.append((current_spot, borodin_grid[current_spot[0]][current_spot[1]]))
            if debug:
                print(f"Borodin Step {step}: {current_spot} -> {borodin_grid[current_spot[0]][current_spot[1]]}")
        else:
            if debug:
                print(f"Borodin Step {step}: {current_spot} -> Out of bounds!")
            break

    return push_block_path_partitions, rsk_path_partitions, borodin_path_partitions

def compare_full_grid_distributions(X, Y, sim_count=1000, debug=False):
    """
    Compare the full grid distributions of push-block, RSK, and Borodin algorithms.
    
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
    
    # Store the first parts for each grid position
    push_block_first_parts = [[[] for _ in range(cols)] for _ in range(rows)]
    rsk_first_parts = [[[] for _ in range(cols)] for _ in range(rows)]
    borodin_first_parts = [[[] for _ in range(cols)] for _ in range(rows)]
    
    print(f"Running {sim_count} simulations on {rows}x{cols} grid...")
    
    for sim in range(sim_count):
        if sim % (sim_count // 10) == 0 and sim > 0:
            print(f"Progress: {sim}/{sim_count} ({100*sim//sim_count}%)")
        
        # Sample all three grids
        push_block_grid = sample_push_block_grid(X, Y)
        rsk_grid = sample_rsk_grid(X, Y)
        borodin_grid = sample_push_block_grid_borodin(X, Y)
        
        # Extract first parts for each position
        for i in range(rows):
            for j in range(cols):
                push_block_first_parts[i][j].append(push_block_grid[i][j].part(1))
                rsk_first_parts[i][j].append(rsk_grid[i][j].part(1))
                borodin_first_parts[i][j].append(borodin_grid[i][j].part(1))
    
    # Compute statistics for each position
    stats = {}
    total_pb_rsk_diff = 0
    total_pb_borodin_diff = 0
    total_rsk_borodin_diff = 0
    max_pb_rsk_diff = 0
    max_pb_borodin_diff = 0
    max_rsk_borodin_diff = 0
    max_pb_rsk_pos = None
    max_pb_borodin_pos = None
    max_rsk_borodin_pos = None
    
    for i in range(rows):
        for j in range(cols):
            push_mean = np.mean(push_block_first_parts[i][j])
            push_std = np.std(push_block_first_parts[i][j])
            rsk_mean = np.mean(rsk_first_parts[i][j])
            rsk_std = np.std(rsk_first_parts[i][j])
            borodin_mean = np.mean(borodin_first_parts[i][j])
            borodin_std = np.std(borodin_first_parts[i][j])
            
            pb_rsk_diff = abs(push_mean - rsk_mean)
            pb_borodin_diff = abs(push_mean - borodin_mean)
            rsk_borodin_diff = abs(rsk_mean - borodin_mean)
            
            total_pb_rsk_diff += pb_rsk_diff
            total_pb_borodin_diff += pb_borodin_diff
            total_rsk_borodin_diff += rsk_borodin_diff
            
            if pb_rsk_diff > max_pb_rsk_diff:
                max_pb_rsk_diff = pb_rsk_diff
                max_pb_rsk_pos = (i, j)
            
            if pb_borodin_diff > max_pb_borodin_diff:
                max_pb_borodin_diff = pb_borodin_diff
                max_pb_borodin_pos = (i, j)
                
            if rsk_borodin_diff > max_rsk_borodin_diff:
                max_rsk_borodin_diff = rsk_borodin_diff
                max_rsk_borodin_pos = (i, j)
            
            stats[(i, j)] = {
                'push_block_mean': push_mean,
                'push_block_std': push_std,
                'rsk_mean': rsk_mean,
                'rsk_std': rsk_std,
                'borodin_mean': borodin_mean,
                'borodin_std': borodin_std,
                'pb_rsk_diff': pb_rsk_diff,
                'pb_borodin_diff': pb_borodin_diff,
                'rsk_borodin_diff': rsk_borodin_diff,
                'push_block_data': push_block_first_parts[i][j],
                'rsk_data': rsk_first_parts[i][j],
                'borodin_data': borodin_first_parts[i][j]
            }
    
    summary = {
        'total_pb_rsk_diff': total_pb_rsk_diff,
        'total_pb_borodin_diff': total_pb_borodin_diff,
        'total_rsk_borodin_diff': total_rsk_borodin_diff,
        'avg_pb_rsk_diff': total_pb_rsk_diff / (rows * cols),
        'avg_pb_borodin_diff': total_pb_borodin_diff / (rows * cols),
        'avg_rsk_borodin_diff': total_rsk_borodin_diff / (rows * cols),
        'max_pb_rsk_diff': max_pb_rsk_diff,
        'max_pb_borodin_diff': max_pb_borodin_diff,
        'max_rsk_borodin_diff': max_rsk_borodin_diff,
        'max_pb_rsk_pos': max_pb_rsk_pos,
        'max_pb_borodin_pos': max_pb_borodin_pos,
        'max_rsk_borodin_pos': max_rsk_borodin_pos,
        'grid_size': (rows, cols),
        'stats': stats
    }
    
    return summary

def plot_grid_comparison(stats_summary, X, Y, save_path=None):
    """
    Plot heatmaps comparing the two algorithms across the full grid.
    
    Parameters:
    - stats_summary: Dictionary returned by compare_full_grid_distributions
    - X, Y: Grid parameters
    - save_path: Optional path to save the plot
    """
    rows, cols = stats_summary['grid_size']
    stats = stats_summary['stats']
    
    # Create matrices for the heatmaps
    push_block_means = np.zeros((rows, cols))
    rsk_means = np.zeros((rows, cols))
    differences = np.zeros((rows, cols))
    
    for i in range(rows):
        for j in range(cols):
            push_block_means[i, j] = stats[(i, j)]['push_block_mean']
            rsk_means[i, j] = stats[(i, j)]['rsk_mean']
            differences[i, j] = stats[(i, j)]['difference']
    
    # Create the plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Push-block heatmap
    im1 = axes[0, 0].imshow(push_block_means, cmap='viridis', aspect='auto')
    axes[0, 0].set_title('Push-Block Algorithm: Mean First Parts')
    axes[0, 0].set_xlabel('Column Index (m)')
    axes[0, 0].set_ylabel('Row Index (n)')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # RSK heatmap
    im2 = axes[0, 1].imshow(rsk_means, cmap='viridis', aspect='auto')
    axes[0, 1].set_title('RSK Algorithm: Mean First Parts')
    axes[0, 1].set_xlabel('Column Index (m)')
    axes[0, 1].set_ylabel('Row Index (n)')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Difference heatmap
    im3 = axes[1, 0].imshow(differences, cmap='Reds', aspect='auto')
    axes[1, 0].set_title('Absolute Difference in Means')
    axes[1, 0].set_xlabel('Column Index (m)')
    axes[1, 0].set_ylabel('Row Index (n)')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # Distribution comparison at max difference position
    max_pos = stats_summary['max_diff_position']
    if max_pos:
        i, j = max_pos
        push_data = stats[(i, j)]['push_block_data']
        rsk_data = stats[(i, j)]['rsk_data']
        
        # Create histograms
        max_val = max(max(push_data) if push_data else 0, max(rsk_data) if rsk_data else 0)
        bins = range(0, max_val + 2)
        
        axes[1, 1].hist(push_data, bins=bins, alpha=0.7, label='Push-Block', density=True)
        axes[1, 1].hist(rsk_data, bins=bins, alpha=0.7, label='RSK', density=True)
        axes[1, 1].set_title(f'Distributions at Max Difference Position ({i}, {j})')
        axes[1, 1].set_xlabel('First Part Value')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle(f'Grid Comparison: X={X}, Y={Y}', fontsize=16, y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()

def print_grid_summary(stats_summary, X, Y):
    """
    Print a summary of the grid comparison results for all three algorithms.
    """
    print(f"\nGrid Comparison Summary")
    print(f"=" * 70)
    print(f"Grid parameters: X={X}, Y={Y}")
    print(f"Grid size: {stats_summary['grid_size'][0]} x {stats_summary['grid_size'][1]}")
    print(f"Total PB-RSK difference: {stats_summary['total_pb_rsk_diff']:.4f}")
    print(f"Total PB-Borodin difference: {stats_summary['total_pb_borodin_diff']:.4f}")
    print(f"Total RSK-Borodin difference: {stats_summary['total_rsk_borodin_diff']:.4f}")
    print(f"Average PB-RSK difference: {stats_summary['avg_pb_rsk_diff']:.4f}")
    print(f"Average PB-Borodin difference: {stats_summary['avg_pb_borodin_diff']:.4f}")
    print(f"Average RSK-Borodin difference: {stats_summary['avg_rsk_borodin_diff']:.4f}")
    print(f"Maximum PB-RSK difference: {stats_summary['max_pb_rsk_diff']:.4f} at {stats_summary['max_pb_rsk_pos']}")
    print(f"Maximum PB-Borodin difference: {stats_summary['max_pb_borodin_diff']:.4f} at {stats_summary['max_pb_borodin_pos']}")
    print(f"Maximum RSK-Borodin difference: {stats_summary['max_rsk_borodin_diff']:.4f} at {stats_summary['max_rsk_borodin_pos']}")
    
    # Show top 5 positions with largest PB-RSK differences
    stats = stats_summary['stats']
    sorted_pb_rsk = sorted(stats.items(), key=lambda x: x[1]['pb_rsk_diff'], reverse=True)
    sorted_pb_borodin = sorted(stats.items(), key=lambda x: x[1]['pb_borodin_diff'], reverse=True)
    sorted_rsk_borodin = sorted(stats.items(), key=lambda x: x[1]['rsk_borodin_diff'], reverse=True)
    
    print(f"\nTop 5 positions with largest PB-RSK differences:")
    print(f"{'Position':<12} {'Push-Block':<12} {'RSK':<12} {'Difference':<12}")
    print(f"{'-'*50}")
    
    for i, ((row, col), data) in enumerate(sorted_pb_rsk[:5]):
        print(f"({row:2d},{col:2d})      {data['push_block_mean']:8.4f}     "
              f"{data['rsk_mean']:8.4f}     {data['pb_rsk_diff']:8.4f}")
              
    print(f"\nTop 5 positions with largest PB-Borodin differences:")
    print(f"{'Position':<12} {'Push-Block':<12} {'Borodin':<12} {'Difference':<12}")
    print(f"{'-'*50}")
    
    for i, ((row, col), data) in enumerate(sorted_pb_borodin[:5]):
        print(f"({row:2d},{col:2d})      {data['push_block_mean']:8.4f}     "
              f"{data['borodin_mean']:8.4f}     {data['pb_borodin_diff']:8.4f}")
              
    print(f"\nTop 5 positions with largest RSK-Borodin differences:")
    print(f"{'Position':<12} {'RSK':<12} {'Borodin':<12} {'Difference':<12}")
    print(f"{'-'*50}")
    
    for i, ((row, col), data) in enumerate(sorted_rsk_borodin[:5]):
        print(f"({row:2d},{col:2d})      {data['rsk_mean']:8.4f}     "
              f"{data['borodin_mean']:8.4f}     {data['rsk_borodin_diff']:8.4f}")

def main():
    parser = argparse.ArgumentParser(
        description="Compare Push-Block and RSK algorithms for Schur processes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare along a specific path
  python schur_testing.py path --path "DRRD" --X 0.5 0.5 --Y 0.5 0.5 --simulations 10000

  # Compare full grid distributions
  python schur_testing.py grid --X 0.3 0.4 0.2 --Y 0.4 0.5 --simulations 1000 --plot

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
                           help='X parameters (space-separated). Default: 0.3 0.4')
    grid_parser.add_argument('--Y', type=float, nargs='+', default=[0.5, 0.5],
                           help='Y parameters (space-separated). Default: 0.4 0.3')
    grid_parser.add_argument('--simulations', type=int, default=1000,
                           help='Number of simulations. Default: 1000')
    grid_parser.add_argument('--plot', action='store_true',
                           help='Show grid comparison heatmaps')
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
    """Run the path-based comparison with all three algorithms."""
    X, Y, path = args.X, args.Y, args.path
    sim_count = args.simulations
    debug = args.debug
    plot = args.plot
    
    print(f"Running path-based comparison:")
    print(f"X={X}, Y={Y}, Path='{path}', Simulations={sim_count}")
    
    # We include the starting position, so we have len(path) + 1 positions
    num_positions = len(path) + 1
    push_block_part1 = [[] for _ in range(num_positions)]
    rsk_part1 = [[] for _ in range(num_positions)]
    borodin_part1 = [[] for _ in range(num_positions)]

    # build up a probability distribution for the three samplers
    for i in range(sim_count):
        if i % (sim_count // 10) == 0 and i > 0:
            print(f"Progress: {i}/{sim_count} ({100*i//sim_count}%)")
            
        push_block_partitions, rsk_partitions, borodin_partitions = sample_all_grids(X, Y, path, debug)
        
        # Make sure we have the expected number of partitions
        min_len = min(len(push_block_partitions), len(rsk_partitions), len(borodin_partitions), num_positions)
        
        for j in range(min_len):
            # Extract the first part (part 1) of each partition
            push_block_partition = push_block_partitions[j][1]  # Get the partition object
            rsk_partition = rsk_partitions[j][1]  # Get the partition object
            borodin_partition = borodin_partitions[j][1]  # Get the partition object
            
            # Get the first part (1-indexed)
            push_block_part1[j].append(push_block_partition.part(1))
            rsk_part1[j].append(rsk_partition.part(1))
            borodin_part1[j].append(borodin_partition.part(1))

    # Display the results as probability distributions
    if plot:
        plot_probability_distributions_three(push_block_part1[:min_len], rsk_part1[:min_len], borodin_part1[:min_len], path)

    # Print summary statistics
    push_block_data = push_block_part1[:min_len]
    rsk_data = rsk_part1[:min_len]
    borodin_data = borodin_part1[:min_len]

    print(f"\nSummary Statistics for Path: {path}")
    print("="*70)
    for step in range(len(path) + 1):
        
        push_mean = np.mean(push_block_data[step])
        push_std = np.std(push_block_data[step])
        rsk_mean = np.mean(rsk_data[step])
        rsk_std = np.std(rsk_data[step])
        borodin_mean = np.mean(borodin_data[step])
        borodin_std = np.std(borodin_data[step])
        
        if step == 0:
            step_label = "Start"
        else:
            step_label = f"{path[step-1]}{step}"
        
        print(f"{step_label}:")
        print(f"  Push-Block: Mean={push_mean:.3f}, Std={push_std:.3f}")
        print(f"  RSK:        Mean={rsk_mean:.3f}, Std={rsk_std:.3f}")
        print(f"  Borodin:    Mean={borodin_mean:.3f}, Std={borodin_std:.3f}")
        print(f"  PB-RSK Diff:     {abs(push_mean - rsk_mean):.3f}")
        print(f"  PB-Borodin Diff: {abs(push_mean - borodin_mean):.3f}")
        print(f"  RSK-Borodin Diff: {abs(rsk_mean - borodin_mean):.3f}")
        print()

def grid_based_comparison(args):
    """Run the new grid-based comparison."""
    X, Y = args.X, args.Y
    sim_count = args.simulations
    debug = args.debug
    plot = args.plot
    save_path = args.save
    
    print(f"Running grid-based comparison:")
    print(f"X={X}, Y={Y}, Simulations={sim_count}")
    
    # Run the comparison
    stats_summary = compare_full_grid_distributions(X, Y, sim_count, debug)
    
    # Print summary
    print_grid_summary(stats_summary, X, Y)
    
    # Show plots if requested
    if plot:
        plot_grid_comparison(stats_summary, X, Y, save_path)

def plot_probability_distributions_three(push_block_data, rsk_data, borodin_data, path):
    """
    Plot probability distributions for the first parts of all three samplers.
    
    Parameters:
    - push_block_data: List of lists containing first parts for each step (push-block sampler)
    - rsk_data: List of lists containing first parts for each step (RSK sampler)
    - borodin_data: List of lists containing first parts for each step (Borodin sampler)
    - path: The path string used for labeling
    """
    num_steps = len(push_block_data)
    
    # Create subplots
    fig, axes = plt.subplots(3, num_steps, figsize=(4*num_steps, 12))
    if num_steps == 1:
        axes = axes.reshape(3, 1)
    
    # Find the overall range for consistent x-axis
    all_values = []
    for step in range(num_steps):
        all_values.extend(push_block_data[step])
        all_values.extend(rsk_data[step])
        all_values.extend(borodin_data[step])
    
    max_value = max(all_values) if all_values else 0
    x_range = range(0, max_value + 2)
    
    for step in range(num_steps):
        # Count frequencies for all samplers
        push_counts = Counter(push_block_data[step])
        push_total = len(push_block_data[step])
        
        rsk_counts = Counter(rsk_data[step])
        rsk_total = len(rsk_data[step])
        
        borodin_counts = Counter(borodin_data[step])
        borodin_total = len(borodin_data[step])
        
        # Calculate probabilities
        push_probs = [push_counts.get(val, 0) / push_total for val in x_range]
        rsk_probs = [rsk_counts.get(val, 0) / rsk_total for val in x_range]
        borodin_probs = [borodin_counts.get(val, 0) / borodin_total for val in x_range]
        
        # Create step label
        if step == 0:
            step_label = "Start"
        else:
            step_label = f"{path[step-1]}{step}"
        
        # Plot push-block distribution
        axes[0, step].bar(x_range, push_probs, alpha=0.7, color='blue', 
                         label='Push-Block')
        axes[0, step].set_title(f'Push-Block: {step_label}')
        axes[0, step].set_xlabel('First Part Value')
        axes[0, step].set_ylabel('Probability')
        axes[0, step].grid(True, alpha=0.3)
        axes[0, step].set_ylim(0, 1)
        
        # Plot RSK distribution
        axes[1, step].bar(x_range, rsk_probs, alpha=0.7, color='red', 
                         label='RSK')
        axes[1, step].set_title(f'RSK: {step_label}')
        axes[1, step].set_xlabel('First Part Value')
        axes[1, step].set_ylabel('Probability')
        axes[1, step].grid(True, alpha=0.3)
        axes[1, step].set_ylim(0, 1)
        
        # Plot Borodin distribution
        axes[2, step].bar(x_range, borodin_probs, alpha=0.7, color='green', 
                         label='Borodin')
        axes[2, step].set_title(f'Borodin: {step_label}')
        axes[2, step].set_xlabel('First Part Value')
        axes[2, step].set_ylabel('Probability')
        axes[2, step].grid(True, alpha=0.3)
        axes[2, step].set_ylim(0, 1)
        
        # Add statistics text
        push_mean = np.mean(push_block_data[step])
        rsk_mean = np.mean(rsk_data[step])
        borodin_mean = np.mean(borodin_data[step])
        
        axes[0, step].text(0.7, 0.9, f'Mean: {push_mean:.2f}', 
                          transform=axes[0, step].transAxes, 
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        axes[1, step].text(0.7, 0.9, f'Mean: {rsk_mean:.2f}', 
                          transform=axes[1, step].transAxes,
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        axes[2, step].text(0.7, 0.9, f'Mean: {borodin_mean:.2f}', 
                          transform=axes[2, step].transAxes,
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.suptitle(f'First Part Probability Distributions Along Path: {path}', 
                 fontsize=16, y=0.95)
    plt.show()


def plot_probability_distributions(push_block_data, rsk_data, path):
    """
    Plot probability distributions for the first parts of both samplers.
    
    Parameters:
    - push_block_data: List of lists containing first parts for each step (push-block sampler)
    - rsk_data: List of lists containing first parts for each step (RSK sampler)
    - path: The path string used for labeling
    """
    num_steps = len(push_block_data)
    
    # Create subplots
    fig, axes = plt.subplots(2, num_steps, figsize=(4*num_steps, 8))
    if num_steps == 1:
        axes = axes.reshape(2, 1)
    
    # Find the overall range for consistent x-axis
    all_values = []
    for step in range(num_steps):
        all_values.extend(push_block_data[step])
        all_values.extend(rsk_data[step])
    
    max_value = max(all_values) if all_values else 0
    x_range = range(0, max_value + 2)
    
    for step in range(num_steps):
        # Count frequencies for push-block sampler
        push_counts = Counter(push_block_data[step])
        push_total = len(push_block_data[step])
        
        # Count frequencies for RSK sampler
        rsk_counts = Counter(rsk_data[step])
        rsk_total = len(rsk_data[step])
        
        # Calculate probabilities
        push_probs = [push_counts.get(val, 0) / push_total for val in x_range]
        rsk_probs = [rsk_counts.get(val, 0) / rsk_total for val in x_range]
        
        # Create step label
        if step == 0:
            step_label = "Start"
        else:
            step_label = f"{path[step-1]}{step}"
        
        # Plot push-block distribution
        axes[0, step].bar(x_range, push_probs, alpha=0.7, color='blue', 
                         label='Push-Block')
        axes[0, step].set_title(f'Push-Block: {step_label}')
        axes[0, step].set_xlabel('First Part Value')
        axes[0, step].set_ylabel('Probability')
        axes[0, step].grid(True, alpha=0.3)
        axes[0, step].set_ylim(0, 1)
        
        # Plot RSK distribution
        axes[1, step].bar(x_range, rsk_probs, alpha=0.7, color='red', 
                         label='RSK')
        axes[1, step].set_title(f'RSK: {step_label}')
        axes[1, step].set_xlabel('First Part Value')
        axes[1, step].set_ylabel('Probability')
        axes[1, step].grid(True, alpha=0.3)
        axes[1, step].set_ylim(0, 1)
        
        # Add statistics text
        push_mean = np.mean(push_block_data[step])
        rsk_mean = np.mean(rsk_data[step])
        
        axes[0, step].text(0.7, 0.9, f'Mean: {push_mean:.2f}', 
                          transform=axes[0, step].transAxes, 
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        axes[1, step].text(0.7, 0.9, f'Mean: {rsk_mean:.2f}', 
                          transform=axes[1, step].transAxes,
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.suptitle(f'First Part Probability Distributions Along Path: {path}', 
                 fontsize=16, y=0.92)
    plt.show()


if __name__ == "__main__":
    main()