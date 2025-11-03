#!/usr/bin/env python3
"""
Script to plot prediction error heatmap for polybench programs.

The heatmap shows relative error (not percentage) defined as:
    |predicted_miss_count - simulated_miss_count| / total_access
    
Positive values (red): over-prediction (predicted > simulated)
Negative values (blue): under-prediction (predicted < simulated)

Y-axis: 30 polybench programs (sorted by name)
X-axis: Different cache configurations:
    - FA-3KB, 12WA-3KB, FA-6KB, 12WA-6KB, ... (grouped by cache size)
    - FA-MEAN, 12WA-MEAN, FA-MEDIAN, 12WA-MEDIAN (statistics across all cache sizes)

Usage: python3 plot-prediction-heatmap.py [--output OUTPUT_FILE]
"""

import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm

# Cache size configurations (in bytes)
# These are common cache sizes that exist in both FA and 12WA simulation databases
CACHE_SIZES = {
    '3KB': 3072,      # 3 KB
    '6KB': 6144,      # 6 KB
    '12KB': 12288,    # 12 KB
    '48KB': 49152,    # 48 KB
}

# Block size in bytes (from the analysis: 8 elements × 8 bytes = 64 bytes)
BLOCK_SIZE_BYTES = 64

def load_data_from_sqlite(db_path):
    """Load data from SQLite database."""
    if not os.path.exists(db_path):
        print(f"Warning: Database '{db_path}' not found")
        return pd.DataFrame()
    
    conn = sqlite3.connect(db_path)
    query = """
    SELECT program, d1_cache_size, d1_miss_count, total_access
    FROM records 
    ORDER BY program, d1_cache_size
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Remove .mlir suffix from program names to match JSON files
    df['program'] = df['program'].str.replace('.mlir', '', regex=False)
    
    return df

def load_total_counts_from_directory(base_dir):
    """Load total counts from base directory JSON files."""
    total_counts = {}
    
    if not os.path.exists(base_dir):
        print(f"Warning: Base directory '{base_dir}' not found for total counts")
        return total_counts
    
    for json_file in os.listdir(base_dir):
        if json_file.endswith('.json'):
            program_name = json_file.replace('.json', '')
            json_path = os.path.join(base_dir, json_file)
            
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    
                # Extract total count from the JSON data
                total_count_str = data.get('total_count', '0')
                
                # Parse the total count (may contain 'R' for symbolic values)
                if ' ' in total_count_str:
                    # Format like "96000000 R" - extract the coefficient
                    total_count = int(total_count_str.split()[0])
                else:
                    total_count = int(total_count_str)
                
                total_counts[program_name] = total_count
                
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                print(f"Warning: Could not parse total count from {json_path}: {e}")
                
    return total_counts

def load_prediction_from_json(json_path):
    """Load prediction data from JSON file with turning points."""
    if not os.path.exists(json_path):
        return None
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        miss_ratio_curve = data.get('miss_ratio_curve', {})
        
        if not miss_ratio_curve:
            return None
            
        turning_points = miss_ratio_curve.get('turning_points', [])
        miss_ratios = miss_ratio_curve.get('miss_ratio', [])
        
        if not turning_points or not miss_ratios:
            return None
        
        # Convert turning points from blocks to bytes
        turning_points_bytes = [tp * BLOCK_SIZE_BYTES for tp in turning_points]
        
        return {
            'turning_points': turning_points_bytes,
            'miss_ratios': miss_ratios
        }
        
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        print(f"Warning: Could not parse {json_path}: {e}")
        return None

def get_prediction_at_cache_size(prediction_data, cache_size_bytes, total_count):
    """
    Get predicted miss count at a specific cache size.
    Uses the nearest turning point (step function - use the value at or before the cache size).
    
    Args:
        prediction_data: Dict with 'turning_points' and 'miss_ratios'
        cache_size_bytes: Cache size in bytes
        total_count: Total number of accesses
    
    Returns:
        Predicted miss count, or None if no prediction available
    """
    if not prediction_data or total_count == 0:
        return None
    
    turning_points = prediction_data['turning_points']
    miss_ratios = prediction_data['miss_ratios']
    
    if not turning_points or not miss_ratios:
        return None
    
    # Find the appropriate miss ratio for this cache size
    # The miss ratio curve is a step function (steps-post in the plot)
    # So we use the miss ratio at the largest turning point <= cache_size
    
    # Find all turning points <= cache_size
    valid_indices = [i for i, tp in enumerate(turning_points) if tp <= cache_size_bytes]
    
    if not valid_indices:
        # Cache size is smaller than the smallest turning point
        # Use the first miss ratio (highest miss rate)
        miss_ratio = miss_ratios[0]
    else:
        # Use the miss ratio at the largest turning point <= cache_size
        idx = valid_indices[-1]
        miss_ratio = miss_ratios[idx]
    
    # Convert miss ratio to miss count
    miss_count = miss_ratio * total_count
    
    return miss_count

def calculate_relative_error(sim_miss_count, pred_miss_count, total_access):
    """Calculate relative error: |predicted - simulated| / total_access"""
    if total_access == 0:
        return np.nan
    
    if pred_miss_count is None or not np.isfinite(pred_miss_count):
        return np.nan
    
    if not np.isfinite(sim_miss_count):
        return np.nan
    
    return abs(pred_miss_count - sim_miss_count) / total_access

def compute_errors_for_program(program, sim_data_fa, sim_data_12wa, pred_data_fa, pred_data_12wa, total_count):
    """
    Compute relative errors for all cache configurations for a single program.
    
    Returns:
        Dictionary with keys: FA-3KB, FA-6KB, FA-12KB, FA-48KB, FA-MEAN, FA-MEDIAN, 
                             12WA-3KB, 12WA-6KB, 12WA-12KB, 12WA-48KB, 12WA-MEAN, 12WA-MEDIAN
    """
    errors = {}
    
    # Process fully associative configurations for selected sizes
    for size_name, cache_size_bytes in CACHE_SIZES.items():
        # Get simulated miss count at this cache size
        sim_row = sim_data_fa[sim_data_fa['d1_cache_size'] == cache_size_bytes]
        
        if sim_row.empty:
            errors[f'FA-{size_name}'] = np.nan
        else:
            sim_miss = sim_row['d1_miss_count'].iloc[0]
            total_access = sim_row['total_access'].iloc[0]
            
            # Get predicted miss count
            pred_miss = get_prediction_at_cache_size(pred_data_fa, cache_size_bytes, total_count)
            
            # Calculate relative error
            error = calculate_relative_error(sim_miss, pred_miss, total_access)
            errors[f'FA-{size_name}'] = error
    
    # Calculate FA mean and median across ALL cache sizes in the database (not just selected ones)
    fa_all_errors = []
    for _, row in sim_data_fa.iterrows():
        cache_size_bytes = row['d1_cache_size']
        sim_miss = row['d1_miss_count']
        total_access = row['total_access']
        
        pred_miss = get_prediction_at_cache_size(pred_data_fa, cache_size_bytes, total_count)
        error = calculate_relative_error(sim_miss, pred_miss, total_access)
        
        if not np.isnan(error):
            fa_all_errors.append(error)
    
    if fa_all_errors:
        errors['FA-MEAN'] = np.mean(fa_all_errors)
        errors['FA-MEDIAN'] = np.median(fa_all_errors)
    else:
        errors['FA-MEAN'] = np.nan
        errors['FA-MEDIAN'] = np.nan
    
    # Process 12-way associative configurations for selected sizes
    for size_name, cache_size_bytes in CACHE_SIZES.items():
        # Get simulated miss count at this cache size
        sim_row = sim_data_12wa[sim_data_12wa['d1_cache_size'] == cache_size_bytes]
        
        if sim_row.empty:
            errors[f'12WA-{size_name}'] = np.nan
        else:
            sim_miss = sim_row['d1_miss_count'].iloc[0]
            total_access = sim_row['total_access'].iloc[0]
            
            # Get predicted miss count
            pred_miss = get_prediction_at_cache_size(pred_data_12wa, cache_size_bytes, total_count)
            
            # Calculate relative error
            error = calculate_relative_error(sim_miss, pred_miss, total_access)
            errors[f'12WA-{size_name}'] = error
    
    # Calculate 12WA mean and median across ALL cache sizes in the database (not just selected ones)
    wa_all_errors = []
    for _, row in sim_data_12wa.iterrows():
        cache_size_bytes = row['d1_cache_size']
        sim_miss = row['d1_miss_count']
        total_access = row['total_access']
        
        pred_miss = get_prediction_at_cache_size(pred_data_12wa, cache_size_bytes, total_count)
        error = calculate_relative_error(sim_miss, pred_miss, total_access)
        
        if not np.isnan(error):
            wa_all_errors.append(error)
    
    if wa_all_errors:
        errors['12WA-MEAN'] = np.mean(wa_all_errors)
        errors['12WA-MEDIAN'] = np.median(wa_all_errors)
    else:
        errors['12WA-MEAN'] = np.nan
        errors['12WA-MEDIAN'] = np.nan
    
    return errors

def main():
    parser = argparse.ArgumentParser(description='Generate prediction error heatmap for polybench')
    parser.add_argument('--output', '-o', default='prediction_error_heatmap.svg',
                       help='Output file name (default: prediction_error_heatmap.svg)')
    parser.add_argument('--sim-fa-db', default='results/fully-associative.db',
                       help='Fully associative simulation database (default: results/fully-associative.db)')
    parser.add_argument('--sim-12wa-db', default='results/12-way.db',
                       help='12-way associative simulation database (default: results/12-way.db)')
    parser.add_argument('--pred-fa-dir', default='results/fully-associative',
                       help='Fully associative prediction directory (default: results/fully-associative)')
    parser.add_argument('--pred-12wa-dir', default='results/12-way',
                       help='12-way associative prediction directory (default: results/12-way)')
    parser.add_argument('--no-pdf', action='store_true',
                       help='Skip PDF conversion with Inkscape')
    
    args = parser.parse_args()
    
    print("Loading simulation data...")
    sim_fa = load_data_from_sqlite(args.sim_fa_db)
    sim_12wa = load_data_from_sqlite(args.sim_12wa_db)
    
    if sim_fa.empty or sim_12wa.empty:
        print("Error: Could not load simulation data")
        sys.exit(1)
    
    print("Loading total counts...")
    total_counts = load_total_counts_from_directory(args.pred_fa_dir)
    
    # Get list of programs (intersection of all datasets)
    programs_fa_sim = set(sim_fa['program'].unique())
    programs_12wa_sim = set(sim_12wa['program'].unique())
    programs_pred = set(total_counts.keys())
    
    programs = sorted(programs_fa_sim & programs_12wa_sim & programs_pred)
    
    print(f"Found {len(programs)} programs in common across all datasets")
    
    if len(programs) == 0:
        print("Error: No common programs found")
        sys.exit(1)
    
    # Define column order - group by cache size instead of by associativity
    columns = [
        'FA-3KB', '12WA-3KB',
        'FA-6KB', '12WA-6KB',
        'FA-12KB', '12WA-12KB',
        'FA-48KB', '12WA-48KB',
        'FA-MEAN', '12WA-MEAN',
        'FA-MEDIAN', '12WA-MEDIAN'
    ]
    
    # Initialize results matrix
    error_matrix = []
    
    print("\nComputing relative errors for each program...")
    for program in tqdm(programs, desc="Processing programs"):
        # Load prediction data
        pred_fa_path = os.path.join(args.pred_fa_dir, f"{program}.json")
        pred_12wa_path = os.path.join(args.pred_12wa_dir, f"{program}.json")
        
        pred_data_fa = load_prediction_from_json(pred_fa_path)
        pred_data_12wa = load_prediction_from_json(pred_12wa_path)
        
        if pred_data_fa is None or pred_data_12wa is None:
            print(f"Warning: Missing prediction data for {program}, skipping")
            continue
        
        # Get total count
        total_count = total_counts.get(program, 0)
        if total_count == 0:
            print(f"Warning: Zero total count for {program}, skipping")
            continue
        
        # Filter simulation data for this program
        sim_data_fa = sim_fa[sim_fa['program'] == program]
        sim_data_12wa = sim_12wa[sim_12wa['program'] == program]
        
        if sim_data_fa.empty or sim_data_12wa.empty:
            print(f"Warning: Missing simulation data for {program}, skipping")
            continue
        
        # Compute errors for all configurations
        errors = compute_errors_for_program(
            program, sim_data_fa, sim_data_12wa,
            pred_data_fa, pred_data_12wa, total_count
        )
        
        # Add to matrix
        row = [program] + [errors.get(col, np.nan) for col in columns]
        error_matrix.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(error_matrix, columns=['Program'] + columns)
    df = df.set_index('Program')
    
    print(f"\nGenerated error matrix with {len(df)} programs")
    
    # Create heatmap
    print("Generating heatmap...")
    fig, ax = plt.subplots(figsize=(12, max(8, len(df) * 0.3)))
    
    # Use 'Wistia' colormap (yellow to pink/purple) without boundaries
    sns.heatmap(df, annot=True, fmt='.4f', cmap='Wistia', 
                cbar_kws={'label': 'Relative Error'},
                linewidths=0, linecolor='none',
                vmin=0, vmax=0.2,
                ax=ax)
    
    # Remove title
    ax.set_xlabel('Cache Configuration', fontsize=12)
    ax.set_ylabel('Program', fontsize=12)
    
    # Rotate x-axis labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    
    plt.tight_layout()
    
    # Save figure as SVG
    plt.savefig(args.output, format='svg', bbox_inches='tight')
    print(f"\nHeatmap saved to: {args.output}")
    
    # Convert SVG to PDF using Inkscape
    if not args.no_pdf:
        import subprocess
        from pathlib import Path
        
        svg_path = Path(args.output)
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
            print("Install Inkscape or use --no-pdf flag to skip this step.")
    
    # Print statistics
    print("\n" + "="*60)
    print("STATISTICS")
    print("="*60)
    
    for col in columns:
        values = df[col].dropna()
        if len(values) > 0:
            print(f"\n{col}:")
            print(f"  Mean:    {values.mean():.6f}")
            print(f"  Median:  {values.median():.6f}")
            print(f"  Min:     {values.min():.6f}")
            print(f"  Max:     {values.max():.6f}")
    
    # Overall statistics
    print(f"\n{'OVERALL (all configurations)'}")
    print("="*60)
    all_values = df.values.flatten()
    all_values = all_values[~np.isnan(all_values)]
    if len(all_values) > 0:
        print(f"Mean:    {all_values.mean():.6f}")
        print(f"Median:  {np.median(all_values):.6f}")
        print(f"Min:     {all_values.min():.6f}")
        print(f"Max:     {all_values.max():.6f}")
    
    print("\nDone!")
    
    # Optionally display the plot
    # plt.show()

if __name__ == "__main__":
    main()
