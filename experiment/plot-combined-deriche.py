#!/usr/bin/env python3
"""
Plot combined deriche miss counts from split loop nests.

This script loads prediction data from deriche1-6 and sums up miss counts
to show the combined behavior. Plots both FA and 12WA predictions, plus
simulation data from the original deriche program.

Usage: python3 plot-combined-deriche.py [output_file]
"""

import matplotlib.pyplot as plt
import sqlite3
import pandas as pd
import sys
import json
import os
import numpy as np
from collections import defaultdict

CACHE_SIZE_48KB = 49152  # 48KB in bytes
BLOCK_SIZE_BYTES = 64
PROGRAM_NAME = 'deriche'

def load_simulation_data():
    """Load simulation data from 12-way2.db for deriche program."""
    db_path = '../results/12-way2.db'
    
    if not os.path.exists(db_path):
        print(f"Warning: Database '{db_path}' not found")
        return pd.DataFrame()
    
    conn = sqlite3.connect(db_path)
    query = """
    SELECT program, d1_cache_size, d1_miss_count, d1_associativity 
    FROM records 
    WHERE program = ? OR program = ?
    ORDER BY d1_cache_size
    """
    df = pd.read_sql_query(query, conn, params=[PROGRAM_NAME, f'{PROGRAM_NAME}.mlir'])
    conn.close()
    
    if df.empty:
        print(f"Warning: No simulation data found for '{PROGRAM_NAME}' in {db_path}")
    
    return df

def load_prediction_from_json(json_path, total_count_override=None):
    """Load prediction data from a single JSON file."""
    if not os.path.exists(json_path):
        return None
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Get total count
        total_count_str = data.get('total_count', '0')
        if ' ' in total_count_str:
            total_count = int(total_count_str.split()[0])
        else:
            total_count = int(total_count_str)
        
        # Use override if provided (for 12WA using FA total count)
        if total_count_override is not None:
            total_count = total_count_override
        
        if total_count == 0:
            print(f"Warning: Zero total count in {json_path}")
            return None
        
        # Get miss ratio curve
        miss_ratio_curve = data.get('miss_ratio_curve', {})
        if not miss_ratio_curve:
            print(f"Warning: No miss_ratio_curve in {json_path}")
            return None
        
        turning_points = miss_ratio_curve.get('turning_points', [])
        miss_ratios = miss_ratio_curve.get('miss_ratio', [])
        
        if not turning_points or not miss_ratios:
            print(f"Warning: Empty miss ratio curve in {json_path}")
            return None
        
        # Convert to cache sizes and miss counts
        cache_sizes = [tp * BLOCK_SIZE_BYTES for tp in turning_points]
        miss_counts = [mr * total_count for mr in miss_ratios]
        
        return {
            'cache_sizes': cache_sizes,
            'miss_counts': miss_counts,
            'total_count': total_count
        }
        
    except Exception as e:
        print(f"Warning: Could not parse {json_path}: {e}")
        return None

def get_miss_count_at_cache_size(pred, cache_size):
    """
    Get miss count at a specific cache size from a prediction.
    Uses binary search for efficiency.
    """
    if not pred:
        return 0
    
    cache_sizes = pred['cache_sizes']
    miss_counts = pred['miss_counts']
    
    # Binary search for the largest cache size <= target
    import bisect
    idx = bisect.bisect_right(cache_sizes, cache_size) - 1
    
    if idx < 0:
        # Cache size is smaller than smallest turning point
        return miss_counts[0]
    else:
        return miss_counts[idx]

def combine_miss_counts(predictions):
    """
    Combine miss counts from multiple predictions.
    Returns a dict mapping cache_size -> total_miss_count.
    Uses only the unique turning points from all predictions.
    """
    if not predictions:
        return {}
    
    # Collect all unique cache sizes (turning points only)
    all_cache_sizes = set()
    for pred in predictions:
        if pred:
            all_cache_sizes.update(pred['cache_sizes'])
    
    all_cache_sizes = sorted(all_cache_sizes)
    
    # For each cache size, sum up miss counts from all predictions
    combined = {}
    
    for cache_size in all_cache_sizes:
        total_miss = sum(get_miss_count_at_cache_size(pred, cache_size) 
                        for pred in predictions if pred)
        combined[cache_size] = total_miss
    
    return combined

def load_combined_predictions():
    """Load and combine predictions from all deriche loop nests."""
    
    # FA predictions (load first to get total counts)
    fa_predictions = []
    fa_total_counts = []
    for i in range(1, 7):
        json_path = os.path.join('results', f'deriche{i}-fa', f'deriche{i}.json')
        pred = load_prediction_from_json(json_path)
        if pred:
            fa_predictions.append(pred)
            fa_total_counts.append(pred['total_count'])
            print(f"Loaded FA prediction for deriche{i} (total_count: {pred['total_count']})")
        else:
            print(f"Warning: Could not load FA prediction for deriche{i}")
            fa_total_counts.append(0)
    
    # 12WA predictions (use FA total counts)
    wa_predictions = []
    for i in range(1, 7):
        json_path = os.path.join('results', f'deriche{i}-12way', f'deriche{i}.json')
        # Use corresponding FA total count
        total_count_override = fa_total_counts[i-1] if i-1 < len(fa_total_counts) else None
        pred = load_prediction_from_json(json_path, total_count_override)
        if pred:
            wa_predictions.append(pred)
            print(f"Loaded 12WA prediction for deriche{i} (using FA total_count: {total_count_override})")
        else:
            print(f"Warning: Could not load 12WA prediction for deriche{i}")
    
    # Combine miss counts
    fa_combined = combine_miss_counts(fa_predictions)
    wa_combined = combine_miss_counts(wa_predictions)
    
    return fa_combined, wa_combined

def create_plot(sim_data, fa_data, wa_data, output_path=None):
    """Create plot for combined deriche predictions."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot simulation data (12-way)
    if not sim_data.empty:
        sim_data_sorted = sim_data.sort_values('d1_cache_size')
        ax.scatter(sim_data_sorted['d1_cache_size'], sim_data_sorted['d1_miss_count'],
                  marker='X', s=150, color='black', alpha=0.8, label='12WA Simulation (Original)',
                  zorder=5)
    
    # Plot FA prediction
    if fa_data:
        cache_sizes = sorted(fa_data.keys())
        miss_counts = [fa_data[cs] for cs in cache_sizes]
        ax.step(cache_sizes, miss_counts,
               where='post', linestyle='--', linewidth=2,
               color='blue', alpha=0.7, label='FA Prediction (Combined)')
    
    # Plot 12WA prediction
    if wa_data:
        cache_sizes = sorted(wa_data.keys())
        miss_counts = [wa_data[cs] for cs in cache_sizes]
        ax.step(cache_sizes, miss_counts,
               where='post', linestyle='--', linewidth=2,
               color='red', alpha=0.7, label='12WA Prediction (Combined)')
    
    # Add vertical line at 48KB
    ax.axvline(x=CACHE_SIZE_48KB, color='purple', linestyle=':', linewidth=2,
              alpha=0.6, label='48KB Cache Size')
    
    # Annotate the vertical line
    ax.text(CACHE_SIZE_48KB, ax.get_ylim()[1] * 0.9, '48KB', 
           rotation=0, verticalalignment='top', horizontalalignment='right',
           fontsize=11, color='purple', fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='purple', alpha=0.7))
    
    # Set labels and title
    ax.set_xlabel('Cache Size (bytes)', fontsize=12)
    ax.set_ylabel('Miss Count', fontsize=12)
    ax.set_title('Combined Deriche Miss Counts (deriche1-6)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Use log scale for x-axis
    ax.set_xscale('log')
    
    # Format y-axis
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # Add legend
    ax.legend(loc='best', fontsize=11)
    
    # Tight layout
    plt.tight_layout()
    
    # Save or show
    if output_path:
        if output_path.endswith('.svg'):
            plt.savefig(output_path, format='svg', bbox_inches='tight')
            print(f"Plot saved to: {output_path}")
            
            # Convert to PDF using Inkscape
            import subprocess
            from pathlib import Path
            
            svg_path = Path(output_path)
            pdf_path = svg_path.with_suffix('.pdf')
            
            try:
                print(f"\nConverting to PDF using Inkscape...")
                subprocess.run([
                    'inkscape',
                    str(svg_path),
                    '--export-filename=' + str(pdf_path),
                    '--export-type=pdf'
                ], check=True, capture_output=True, text=True)
                print(f"PDF saved to: {pdf_path}")
            except subprocess.CalledProcessError as e:
                print(f"Warning: Failed to convert to PDF: {e}")
            except FileNotFoundError:
                print("Warning: Inkscape not found. Skipping PDF conversion.")
        else:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {output_path}")
    else:
        plt.show()
    
    return fig

def main():
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        print("Usage: python3 plot-combined-deriche.py [output_file]")
        print()
        print("Arguments:")
        print("  output_file   Optional output file name (default: combined_deriche.svg)")
        print()
        print("Examples:")
        print("  python3 plot-combined-deriche.py")
        print("  python3 plot-combined-deriche.py combined_deriche.svg")
        print()
        print("The script plots combined deriche miss counts from deriche1-6")
        print("for both FA and 12WA predictions, along with original simulation data.")
        sys.exit(0)
    
    output_file = sys.argv[1] if len(sys.argv) > 1 else 'combined_deriche.svg'
    
    print("Loading simulation data from original deriche...")
    sim_data = load_simulation_data()
    if not sim_data.empty:
        print(f"Found {len(sim_data)} simulation points")
    
    print("\nLoading predictions from deriche1-6...")
    print()
    
    # Load and combine predictions
    fa_data, wa_data = load_combined_predictions()
    
    if not fa_data and not wa_data:
        print("\nError: No prediction data found")
        print("Make sure to run run-prediction.sh first")
        sys.exit(1)
    
    print()
    print(f"FA combined: {len(fa_data)} cache size points")
    print(f"12WA combined: {len(wa_data)} cache size points")
    
    # Create plot
    print("\nCreating plot...")
    create_plot(sim_data, fa_data, wa_data, output_file)
    
    print("Done!")

if __name__ == "__main__":
    main()
