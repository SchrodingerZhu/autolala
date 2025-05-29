#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import numpy as np
import argparse
import matplotlib.pyplot as plt

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

def run(matmul_json, salt_json, svg_output):
    # Load data from JSON files
    with open(matmul_json) as f:
        matmul_data = json.load(f)
        
    with open(salt_json) as f:
        salt_data = json.load(f)

    # Extract data for matmul-t2
    matmul_blocks = np.array(matmul_data['blocks'])
    matmul_miss_ratio = np.array(matmul_data['miss_ratio'])

    # Extract data for salt step function
    salt_turning_points = np.array(salt_data['miss_ratio_curve']['turning_points'])
    salt_miss_ratio = np.array(salt_data['miss_ratio_curve']['miss_ratio'])


    # Create interpolated predictions using salt data
    salt_interp = np.interp(matmul_blocks, salt_turning_points, salt_miss_ratio,
                        left=1.0, right=0.0)
    salt_predictions = salt_interp

    # Calculate Mean Squared Error
    
    # compute mean absolute percentage error only where both arrays have values
    valid_mask = np.arange(len(salt_predictions)) < len(matmul_miss_ratio)
    mse = np.mean((matmul_miss_ratio[valid_mask] - salt_predictions[valid_mask]) ** 2)
    mape = np.mean(np.abs((matmul_miss_ratio[valid_mask] - salt_predictions[valid_mask]) / matmul_miss_ratio[valid_mask])) * 100
    print(f'MAPE: {mape:.2f}%')

    # Create the plot
    plt.figure(figsize=(12, 6))

    plt.plot(matmul_blocks[valid_mask], matmul_miss_ratio[valid_mask], 
            label='Cachegrind Simulation', alpha=0.7)

    # Plot salt step function (cut short where matmul data is missing)
    plt.step(salt_turning_points, salt_miss_ratio, 
            where='post', label='SALT Prediction', linestyle='--')

    # Formatting
    plt.xlabel('Cache Size (#Blocks)', fontsize=30)
    plt.ylabel('Miss Ratio', fontsize=30)
    plt.legend(fontsize=18, markerscale=2)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.xscale('log')
    plt.yscale('log')

    # Increase tick label size
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tick_params(axis='both', which='minor', labelsize=16)

    # Save and show plot
    plt.tight_layout()
    print(f'MSE: {mse:.2e}')
    plt.savefig(svg_output, format='svg')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark Graph Plotter')
    parser.add_argument('--matmul-json', type=str, required=True, help='Path to matmul-t2 JSON file')
    parser.add_argument('--salt-json', type=str, required=True, help='Path to salt step function JSON file')
    parser.add_argument('--svg-output', type=str, required=True, help='Output SVG file path')

    args = parser.parse_args()
    
    run(args.matmul_json, args.salt_json, args.svg_output)
