#!/usr/bin/env python3
"""
Ad-hoc script to plot deriche miss counts.
Plots simulation data from 12-way.db only and all prediction data.
Includes a vertical line at 48KB cache size.

Usage: python3 plot-deriche.py [output_file]
       python3 plot-deriche.py
       python3 plot-deriche.py deriche_plot.png
"""

import sqlite3
import matplotlib.pyplot as plt
import pandas as pd
import sys
import json
import os
import numpy as np

PROGRAM_NAME = 'deriche'
CACHE_SIZE_48KB = 49152  # 48KB in bytes

def load_simulation_data():
    """Load simulation data from 12-way2.db for deriche program."""
    db_path = 'results/12-way2.db'
    
    if not os.path.exists(db_path):
        print(f"Error: Database '{db_path}' not found")
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

def get_total_count_from_fa():
    """Get total count from fully-associative directory."""
    fa_json_path = os.path.join('results/fully-associative', f'{PROGRAM_NAME}.json')
    
    if not os.path.exists(fa_json_path):
        return 0
    
    try:
        with open(fa_json_path, 'r') as f:
            data = json.load(f)
        
        total_count_str = data.get('total_count', '0')
        if ' ' in total_count_str:
            return int(total_count_str.split()[0])
        else:
            return int(total_count_str)
    except (json.JSONDecodeError, ValueError, KeyError):
        return 0

def load_prediction_data():
    """Load all prediction data from JSON directories."""
    # Get total count from FA directory (used as fallback for 12WA predictions)
    fa_total_count = get_total_count_from_fa()
    
    prediction_dirs = [
        ('FA Prediction', 'results/fully-associative', 'blue', '--', None),
        ('12WA Prediction', 'results/12-way', 'red', '--', fa_total_count),
        ('12WA Skew-1.5 Prediction', 'results/12-way-skew-1.5', 'green', '--', fa_total_count),
    ]
    
    all_predictions = []
    
    for name, dir_path, color, linestyle, fallback_total in prediction_dirs:
        json_path = os.path.join(dir_path, f'{PROGRAM_NAME}.json')
        
        if not os.path.exists(json_path):
            print(f"Warning: Prediction file not found: {json_path}")
            continue
        
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Get total count (use fallback if provided and needed)
            total_count_str = data.get('total_count', '0')
            if ' ' in total_count_str:
                total_count = int(total_count_str.split()[0])
            else:
                total_count = int(total_count_str)
            
            # Use fallback if total count is zero
            if total_count == 0 and fallback_total:
                total_count = fallback_total
            
            if total_count == 0:
                print(f"Warning: Zero total count in {json_path}")
                continue
            
            # Get miss ratio curve
            miss_ratio_curve = data.get('miss_ratio_curve', {})
            if not miss_ratio_curve:
                print(f"Warning: No miss_ratio_curve in {json_path}")
                continue
            
            turning_points = miss_ratio_curve.get('turning_points', [])
            miss_ratios = miss_ratio_curve.get('miss_ratio', [])
            
            if not turning_points or not miss_ratios:
                print(f"Warning: Empty miss ratio curve in {json_path}")
                continue
            
            # Convert to cache sizes and miss counts
            cache_sizes = [tp * 64 for tp in turning_points]  # Convert blocks to bytes
            miss_counts = [mr * total_count for mr in miss_ratios]
            
            all_predictions.append({
                'name': name,
                'cache_sizes': cache_sizes,
                'miss_counts': miss_counts,
                'color': color,
                'linestyle': linestyle
            })
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"Warning: Could not parse {json_path}: {e}")
            continue
    
    return all_predictions

def create_plot(sim_data, pred_data, output_path=None):
    """Create plot for deriche program."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot simulation data (12-way only)
    if not sim_data.empty:
        sim_data_sorted = sim_data.sort_values('d1_cache_size')
        ax.scatter(sim_data_sorted['d1_cache_size'], sim_data_sorted['d1_miss_count'],
                  marker='X', s=150, color='black', alpha=0.8, label='12WA Simulation',
                  zorder=5)
    
    # Plot all prediction data
    for pred in pred_data:
        ax.step(pred['cache_sizes'], pred['miss_counts'],
               where='post', linestyle=pred['linestyle'], linewidth=2,
               color=pred['color'], alpha=0.7, label=pred['name'])
    
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
        # Determine format from extension
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
                print("Make sure Inkscape is installed and in your PATH")
            except FileNotFoundError:
                print("Warning: Inkscape not found. Skipping PDF conversion.")
                print("Install Inkscape or manually convert SVG to PDF.")
        else:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {output_path}")
    else:
        plt.show()
    
    return fig

def main():
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        print("Usage: python3 plot-deriche.py [output_file]")
        print()
        print("Arguments:")
        print("  output_file   Optional output file name (default: deriche_plot.svg)")
        print("                Use .svg extension for SVG output with automatic PDF conversion")
        print()
        print("Examples:")
        print("  python3 plot-deriche.py")
        print("  python3 plot-deriche.py deriche_plot.svg")
        print("  python3 plot-deriche.py deriche_plot.png")
        print()
        print("The script plots deriche miss counts with:")
        print("  - Simulation data from 12-way2.db only")
        print("  - All prediction data from JSON directories")
        print("  - Vertical line at 48KB cache size")
        sys.exit(0)
    
    output_file = sys.argv[1] if len(sys.argv) > 1 else 'deriche_plot.svg'
    
    print(f"Loading data for '{PROGRAM_NAME}'...")
    
    # Load simulation data
    sim_data = load_simulation_data()
    
    # Load prediction data
    pred_data = load_prediction_data()
    
    if sim_data.empty and not pred_data:
        print("Error: No data found to plot")
        sys.exit(1)
    
    print(f"Found {len(sim_data)} simulation points")
    print(f"Found {len(pred_data)} prediction datasets")
    
    # Create plot
    print("Creating plot...")
    create_plot(sim_data, pred_data, output_file)
    
    print("Done!")

if __name__ == "__main__":
    main()
