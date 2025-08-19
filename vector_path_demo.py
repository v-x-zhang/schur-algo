#!/usr/bin/env python3
"""
Demo script showing different vector-style path rendering options
for the Schur process visualization.
"""

import matplotlib.pyplot as plt
import numpy as np

def demo_vector_styles():
    """Demonstrate different vector path rendering styles."""
    
    # Create a simple 5x5 grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Vector Path Rendering Options', fontsize=16, fontweight='bold')
    
    # Sample path coordinates
    path_x = [0.5, 1.5, 2.5, 2.5, 3.5, 4.5]
    path_y = [4.5, 3.5, 2.5, 1.5, 0.5, 0.5]
    
    styles = [
        ('Vector (Thin Black)', 'k-', 1, 3),
        ('Bold Black', 'k-', 2, 6), 
        ('Dashed Vector', 'k--', 1, 3),
        ('Arrows', 'k-', 1, 3)
    ]
    
    for idx, (title, linestyle, linewidth, markersize) in enumerate(styles):
        ax = axes[idx // 2, idx % 2]
        
        # White background grid
        for i in range(5):
            for j in range(5):
                rect = plt.Rectangle((i, j), 1, 1, facecolor='white', 
                                   edgecolor='black', linewidth=0.5)
                ax.add_patch(rect)
        
        # Draw path
        if idx == 3:  # Arrows style
            ax.plot(path_x, path_y, linestyle, linewidth=linewidth, alpha=1.0)
            # Add arrows
            for i in range(len(path_x) - 1):
                dx = path_x[i+1] - path_x[i]
                dy = path_y[i+1] - path_y[i]
                ax.annotate('', xy=(path_x[i+1], path_y[i+1]), 
                           xytext=(path_x[i], path_y[i]),
                           arrowprops=dict(arrowstyle='->', color='black', lw=1))
        else:
            ax.plot(path_x, path_y, linestyle, linewidth=linewidth, alpha=1.0)
        
        # Start/end markers
        ax.plot(path_x[0], path_y[0], 'ko', markersize=markersize, 
               markerfacecolor='white', markeredgecolor='black', markeredgewidth=1)
        ax.plot(path_x[-1], path_y[-1], 'ks', markersize=markersize, 
               markerfacecolor='white', markeredgecolor='black', markeredgewidth=1)
        
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 5)
        ax.set_aspect('equal')
        ax.set_title(title, fontweight='bold')
        ax.set_xticks(range(6))
        ax.set_yticks(range(6))
        ax.grid(False)
    
    plt.tight_layout()
    plt.savefig('vector_path_styles.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()

def demo_background_options():
    """Demonstrate different background rendering options."""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Background Style Options for Vector Paths', fontsize=16, fontweight='bold')
    
    # Sample data (partition sizes)
    grid_data = np.array([
        [0, 1, 2, 1, 0],
        [1, 2, 3, 2, 1], 
        [2, 3, 4, 3, 2],
        [1, 2, 3, 2, 1],
        [0, 1, 2, 1, 0]
    ])
    
    path_x = [0.5, 1.5, 2.5, 3.5, 4.5]
    path_y = [4.5, 3.5, 2.5, 1.5, 0.5]
    
    backgrounds = [
        ('Pure White', 'white'),
        ('Grayscale', 'grayscale'),
        ('Colored', 'colored')
    ]
    
    for idx, (title, bg_type) in enumerate(backgrounds):
        ax = axes[idx]
        
        # Draw grid with different background styles
        for i in range(5):
            for j in range(5):
                value = grid_data[j, i]
                
                if bg_type == 'white':
                    color = 'white'
                elif bg_type == 'grayscale':
                    gray_val = 0.9 - 0.2 * (value / 4.0) if value > 0 else 1.0
                    color = (gray_val, gray_val, gray_val)
                else:  # colored
                    if value == 0:
                        color = 'lightgray'
                    else:
                        color = plt.cm.Blues(0.3 + 0.7 * (value / 4.0))
                
                linewidth = 0.5 if bg_type == 'white' else 1.0
                rect = plt.Rectangle((i, j), 1, 1, facecolor=color, 
                                   edgecolor='black', linewidth=linewidth)
                ax.add_patch(rect)
        
        # Draw vector path (thin black line)
        ax.plot(path_x, path_y, 'k-', linewidth=1, alpha=1.0)
        ax.plot(path_x[0], path_y[0], 'ko', markersize=3, 
               markerfacecolor='white', markeredgecolor='black', markeredgewidth=1)
        ax.plot(path_x[-1], path_y[-1], 'ks', markersize=3, 
               markerfacecolor='white', markeredgecolor='black', markeredgewidth=1)
        
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 5)
        ax.set_aspect('equal')
        ax.set_title(title, fontweight='bold')
        ax.set_xticks(range(6))
        ax.set_yticks(range(6))
        ax.grid(False)
    
    plt.tight_layout()
    plt.savefig('vector_background_styles.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()

if __name__ == "__main__":
    print("Generating vector path style demonstrations...")
    demo_vector_styles()
    demo_background_options()
    print("Demo images saved as 'vector_path_styles.png' and 'vector_background_styles.png'")
