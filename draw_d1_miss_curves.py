#!/usr/bin/env python3
"""
Script to draw D1 miss count curves for programs listed in target/list.txt
Uses data from the fully-associative cache database where d1_associativity represents cache size in blocks.
"""

import sqlite3
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

def read_program_list(list_file):
    """Read the list of programs from the list.txt file."""
    programs = []
    with open(list_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                # Convert path format to match database entries
                program_name = os.path.basename(line) + '.mlir'
                programs.append(program_name)
    return programs

def fetch_data(db_path, program):
    """Fetch D1 miss count data for a specific program."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    query = """
    SELECT d1_associativity, d1_miss_count 
    FROM records 
    WHERE program = ? 
    ORDER BY d1_associativity
    """
    
    cursor.execute(query, (program,))
    data = cursor.fetchall()
    conn.close()
    
    if not data:
        print(f"Warning: No data found for program '{program}'")
        return [], []
    
    cache_sizes, miss_counts = zip(*data)
    return list(cache_sizes), list(miss_counts)

def plot_miss_curves(programs_data, output_file=None):
    """Plot D1 miss count curves for all programs."""
    plt.figure(figsize=(12, 8))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    markers = ['o', 's', '^', 'D', 'v', '<']
    
    for i, (program_name, cache_sizes, miss_counts) in enumerate(programs_data):
        if not cache_sizes:
            continue
            
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        
        # Clean program name for legend
        clean_name = program_name.replace('.mlir', '').replace('_', ' ').title()
        
        plt.plot(cache_sizes, miss_counts, 
                marker=marker, 
                color=color, 
                linewidth=2, 
                markersize=6,
                label=clean_name,
                alpha=0.8,
                drawstyle='steps-post')
    
    plt.xlabel('Cache Size (blocks)', fontsize=12)
    plt.ylabel('D1 Miss Count', fontsize=12)
    plt.title('D1 Miss Count vs Cache Size (Fully Associative)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    # Use log scale for better visualization if the range is large
    if any(miss_counts for _, _, miss_counts in programs_data if miss_counts):
        max_miss = max(max(miss_counts) for _, _, miss_counts in programs_data if miss_counts)
        non_zero_miss_counts = [mc for _, _, miss_counts in programs_data if miss_counts for mc in miss_counts if mc > 0]
        if non_zero_miss_counts:
            min_miss = min(non_zero_miss_counts)
            if max_miss / min_miss > 100:
                plt.yscale('log')
                plt.xscale('log')
                plt.ylabel('D1 Miss Count (log scale)', fontsize=12)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
    
    plt.show()

def print_data_summary(programs_data):
    """Print a summary of the data."""
    print("\n" + "="*60)
    print("DATA SUMMARY")
    print("="*60)
    
    for program_name, cache_sizes, miss_counts in programs_data:
        print(f"\nProgram: {program_name}")
        if cache_sizes:
            print(f"  Cache size range: {min(cache_sizes)} - {max(cache_sizes)} blocks")
            print(f"  Miss count range: {min(miss_counts):,} - {max(miss_counts):,}")
            print(f"  Data points: {len(cache_sizes)}")
        else:
            print("  No data available")

def main():
    # File paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    list_file = os.path.join(script_dir, 'target', 'list.txt')
    db_path = '/home/schrodingerzy/Documents/contractions/data-fully-associative-matrix.db'
    output_file = os.path.join(script_dir, 'd1_miss_curves.png')
    
    # Check if required files exist
    if not os.path.exists(list_file):
        print(f"Error: List file not found: {list_file}")
        sys.exit(1)
    
    if not os.path.exists(db_path):
        print(f"Error: Database file not found: {db_path}")
        sys.exit(1)
    
    # Read program list
    try:
        programs = read_program_list(list_file)
        print(f"Programs to analyze: {programs}")
    except Exception as e:
        print(f"Error reading program list: {e}")
        sys.exit(1)
    
    # Fetch data for all programs
    programs_data = []
    for program in programs:
        try:
            cache_sizes, miss_counts = fetch_data(db_path, program)
            programs_data.append((program, cache_sizes, miss_counts))
            print(f"Loaded {len(cache_sizes)} data points for {program}")
        except Exception as e:
            print(f"Error fetching data for {program}: {e}")
            programs_data.append((program, [], []))
    
    # Print data summary
    print_data_summary(programs_data)
    
    # Plot the curves
    if any(cache_sizes for _, cache_sizes, _ in programs_data):
        try:
            plot_miss_curves(programs_data, output_file)
        except Exception as e:
            print(f"Error creating plot: {e}")
            sys.exit(1)
    else:
        print("No data available for plotting.")
        sys.exit(1)

if __name__ == "__main__":
    main()