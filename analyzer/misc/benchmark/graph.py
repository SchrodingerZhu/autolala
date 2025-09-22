#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import numpy as np
import argparse
import matplotlib.pyplot as plt

# Set the plot style to ggplot
plt.style.use('ggplot')

def binary_search_interp_vectorized(matmul_blocks, salt_turning_points, salt_miss_ratio):
    # Convert inputs to numpy arrays if they aren't already
    matmul_blocks = np.asarray(matmul_blocks)
    salt_turning_points = np.asarray(salt_turning_points)
    salt_miss_ratio = np.asarray(salt_miss_ratio)
    
    # Initialize result array
    result = np.zeros_like(matmul_blocks)
    
    # Handle empty turning points case
    if len(salt_turning_points) == 0:
        return result
    
    # Get the indices for each block
    # Using searchsorted which is numpy's equivalent of bisect
    indices = np.searchsorted(salt_turning_points, matmul_blocks, side='right') - 1
    
    # Handle values before first turning point
    mask_before = matmul_blocks < salt_turning_points[0]
    result[mask_before] = 1.0
    
    # Handle values after last turning point
    mask_after = matmul_blocks >= salt_turning_points[-1]
    result[mask_after] = 0.0
    
    # Handle values in between
    mask_middle = ~mask_before & ~mask_after
    valid_indices = indices[mask_middle]
    # Ensure indices are within bounds (should be already due to masks)
    valid_indices = np.clip(valid_indices, 0, len(salt_miss_ratio) - 1)
    result[mask_middle] = salt_miss_ratio[valid_indices]
    
    return result

def run(svg_output):
    # Define the file groups with ggplot-friendly colors
    groups = [
        {
            'matmul_file': 'matmul.json',
            'salt_file': 'matmul-salt.json',
            'label': 'No Tiling',
            'color': '#1f77b4'  # ggplot blue
        },
        {
            'matmul_file': 'matmul-t1.json',
            'salt_file': 'matmul-t1-salt.json',
            'label': 'T1 (Tiled Once)',
            'color': '#ff7f0e'  # ggplot orange
        },
        {
            'matmul_file': 'matmul-t2.json',
            'salt_file': 'matmul-t2-salt.json',
            'label': 'T2 (Tiled Twice)',
            'color': '#2ca02c'  # ggplot green
        }
    ]
    
    # Create the plot
    plt.figure(figsize=(14, 8))
    
    for group in groups:
        # Load data from JSON files
        with open(group['matmul_file']) as f:
            matmul_data = json.load(f)
            
        with open(group['salt_file']) as f:
            salt_data = json.load(f)

        # Extract data for matmul
        matmul_blocks = np.array(matmul_data['blocks'])
        matmul_miss_ratio = np.array(matmul_data['miss_ratio'])

        # Extract data for salt step function
        salt_turning_points = np.array(salt_data['miss_ratio_curve']['turning_points'])
        salt_miss_ratio = np.array(salt_data['miss_ratio_curve']['miss_ratio'])

        # For comparison metrics, we still need to interpolate the SALT predictions at matmul points
        salt_predictions = np.interp(matmul_blocks, salt_turning_points, salt_miss_ratio,
                                   left=1.0, right=0.0)

        # Calculate Mean Squared Error
        # compute mean absolute percentage error only where both arrays have values
        valid_mask = np.arange(len(salt_predictions)) < len(matmul_miss_ratio)
        mse = np.mean((matmul_miss_ratio[valid_mask] - salt_predictions[valid_mask]) ** 2)
        mape = np.mean(np.abs((matmul_miss_ratio[valid_mask] - salt_predictions[valid_mask]) / matmul_miss_ratio[valid_mask])) * 100
        
        print(f'{group["label"]} - MAPE: {mape:.2f}%, MSE: {mse:.2e}')

        # Plot cachegrind simulation
        plt.plot(matmul_blocks[valid_mask], matmul_miss_ratio[valid_mask], 
                label=f'Cachegrind - {group["label"]}', 
                color=group['color'], alpha=0.8, linewidth=2.5, marker='o', 
                markersize=3, markevery=max(1, len(matmul_blocks[valid_mask]) // 20))

        # Plot salt step function using matplotlib's step function
        if len(salt_turning_points) > 0 and len(salt_miss_ratio) > 0:
            # Extend the last value to the right boundary of the plot
            # Find the maximum x-value in the current plot to extend to
            plot_max_x = max(matmul_blocks.max(), salt_turning_points.max()) * 2
            
            # Add one more point to extend the final value
            extended_turning_points = np.append(salt_turning_points, plot_max_x)
            extended_miss_ratio = np.append(salt_miss_ratio, salt_miss_ratio[-1])
            
            plt.step(extended_turning_points, extended_miss_ratio, 
                    where='post', label=f'SALT - {group["label"]}', 
                    color=group['color'], linestyle='--', linewidth=2.5, alpha=0.9)

    # Formatting with ggplot style
    plt.xlabel('Cache Size (#Blocks)', fontsize=24, fontweight='bold')
    plt.ylabel('Miss Ratio', fontsize=24, fontweight='bold')
    plt.legend(fontsize=18, frameon=True, fancybox=True, shadow=True, loc='upper right', markerscale=1.5)
    plt.xscale('log')
    plt.yscale('log')

    # Increase tick label size
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tick_params(axis='both', which='minor', labelsize=12)

    # Save and show plot
    plt.tight_layout()
    plt.savefig(svg_output, format='svg')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark Graph Plotter - Multiple Groups Comparison')
    parser.add_argument('--svg-output', type=str, required=True, help='Output SVG file path')

    args = parser.parse_args()
    
    run(args.svg_output)
