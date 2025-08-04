from schur_sampler import sample_push_block_grid, sample_rsk_grid
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

def sample_both_grids(X, Y, path, debug):
    """
    Sample both push block and RSK grids based on the provided path.
    
    Parameters:
    - X: List of probabilities for each row in the grid.
    - Y: List of probabilities for each column in the grid.
    - path: The path to follow in the grid (a string of 'D' and 'R' characters).
    - debug: Boolean flag to enable debug mode.
    Returns:
    - the push block path partitions and the RSK path partitions.
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
        print(f"Start: {current_spot} -> {push_block_grid[current_spot[0]][current_spot[1]]}")

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
                print(f"Step {step}: {current_spot} -> {push_block_grid[current_spot[0]][current_spot[1]]}")
        else:
            if debug:
                print(f"Step {step}: {current_spot} -> Out of bounds!")
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

def main():
    X = [0.5, 0.5]
    Y = [0.5, 0.5]
    path = "DRRD"
    plot = True
    debug = False
    sim_count = 10000
    
    # We include the starting position, so we have len(path) + 1 positions
    num_positions = len(path) + 1
    push_block_part1 = [[] for _ in range(num_positions)]
    rsk_part1 = [[] for _ in range(num_positions)]

    # build up a probability distribution for the two samplers
    for i in range(sim_count):
        push_block_partitions, rsk_partitions = sample_both_grids(X, Y, path, debug)
        
        # Make sure we have the expected number of partitions
        min_len = min(len(push_block_partitions), len(rsk_partitions), num_positions)
        
        for j in range(min_len):
            # Extract the first part (part 1) of each partition
            push_block_partition = push_block_partitions[j][1]  # Get the partition object
            rsk_partition = rsk_partitions[j][1]  # Get the partition object
            
            # Get the first part (1-indexed)
            push_block_part1[j].append(push_block_partition.part(1))
            rsk_part1[j].append(rsk_partition.part(1))

    # Display the results as probability distributions
    if plot:
        plot_probability_distributions(push_block_part1[:min_len], rsk_part1[:min_len], path)

    # Print summary statistics
    push_block_data = push_block_part1[:min_len]
    rsk_data = rsk_part1[:min_len]

    print(f"\nSummary Statistics for Path: {path}")
    print("="*50)
    for step in range(len(path) + 1):
        
        push_mean = np.mean(push_block_data[step])
        push_std = np.std(push_block_data[step])
        rsk_mean = np.mean(rsk_data[step])
        rsk_std = np.std(rsk_data[step])
        
        if step == 0:
            step_label = "Start"
        else:
            step_label = f"{path[step-1]}{step}"
        
        print(f"{step_label}:")
        print(f"  Push-Block: Mean={push_mean:.3f}, Std={push_std:.3f}")
        print(f"  RSK:        Mean={rsk_mean:.3f}, Std={rsk_std:.3f}")
        print(f"  Difference: {abs(push_mean - rsk_mean):.3f}")
        print()

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