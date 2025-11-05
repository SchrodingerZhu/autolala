#!/usr/bin/env python3

"""
Script to plot comparison heatmap for polybench programs.

The heatmap shows 4 columns for each cache size:
    FA-xK-SIM: FA simulated miss ratio
    FA-xK-PRED: FA predicted miss ratio
    12WA-xK-SIM: 12WA simulated miss ratio
    12WA-xK-PRED: 12WA predicted miss ratio

Each pair (SIM, PRED) is colored by relative error = |simulation - prediction| / simulation
Also includes mean and median statistics for each type.

Y-axis: polybench programs (sorted by name)
X-axis: 16 columns (4 columns × 4 cache sizes) + statistics

Usage: python3 plot-comparision-heatmap.py [--output OUTPUT_FILE]
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

CACHE_SIZES = {
    '3K': 3072,
    '6K': 6144,
    '12K': 12288,
    '48K': 49152,
}
BLOCK_SIZE_BYTES = 64

def load_data_from_sqlite(db_path):
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
    df['program'] = df['program'].str.replace('.mlir', '', regex=False)
    return df

def load_total_counts_from_directory(base_dir):
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
                total_count_str = data.get('total_count', '0')
                if ' ' in total_count_str:
                    total_count = int(total_count_str.split()[0])
                else:
                    total_count = int(total_count_str)
                total_counts[program_name] = total_count
            except Exception as e:
                print(f"Warning: Could not parse total count from {json_path}: {e}")
    return total_counts

def load_prediction_from_json(json_path):
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
        turning_points_bytes = [tp * BLOCK_SIZE_BYTES for tp in turning_points]
        return {
            'turning_points': turning_points_bytes,
            'miss_ratios': miss_ratios
        }
    except Exception as e:
        print(f"Warning: Could not parse {json_path}: {e}")
        return None

def get_prediction_miss_ratio(prediction_data, cache_size_bytes):
    if not prediction_data:
        return np.nan
    turning_points = prediction_data['turning_points']
    miss_ratios = prediction_data['miss_ratios']
    if not turning_points or not miss_ratios:
        return np.nan
    valid_indices = [i for i, tp in enumerate(turning_points) if tp <= cache_size_bytes]
    if not valid_indices:
        miss_ratio = miss_ratios[0]
    else:
        idx = valid_indices[-1]
        miss_ratio = miss_ratios[idx]
    return miss_ratio

# Ad-hoc functions for combining deriche1-6 predictions
def load_prediction_from_json_with_total(json_path, total_count_override=None):
    """Load prediction data with total count."""
    if not os.path.exists(json_path):
        return None
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        total_count_str = data.get('total_count', '0')
        if ' ' in total_count_str:
            total_count = int(total_count_str.split()[0])
        else:
            total_count = int(total_count_str)
        
        if total_count_override is not None:
            total_count = total_count_override
        
        if total_count == 0:
            return None
        
        miss_ratio_curve = data.get('miss_ratio_curve', {})
        if not miss_ratio_curve:
            return None
        
        turning_points = miss_ratio_curve.get('turning_points', [])
        miss_ratios = miss_ratio_curve.get('miss_ratio', [])
        
        if not turning_points or not miss_ratios:
            return None
        
        cache_sizes = [tp * BLOCK_SIZE_BYTES for tp in turning_points]
        miss_counts = [mr * total_count for mr in miss_ratios]
        
        return {
            'cache_sizes': cache_sizes,
            'miss_counts': miss_counts,
            'total_count': total_count
        }
    except Exception as e:
        return None

def get_miss_count_at_cache_size(pred, cache_size):
    """Get miss count at a specific cache size from a prediction."""
    if not pred:
        return 0
    
    cache_sizes = pred['cache_sizes']
    miss_counts = pred['miss_counts']
    
    import bisect
    idx = bisect.bisect_right(cache_sizes, cache_size) - 1
    
    if idx < 0:
        return miss_counts[0]
    else:
        return miss_counts[idx]

def combine_miss_counts(predictions):
    """Combine miss counts from multiple predictions."""
    if not predictions:
        return {}
    
    all_cache_sizes = set()
    for pred in predictions:
        if pred:
            all_cache_sizes.update(pred['cache_sizes'])
    
    all_cache_sizes = sorted(all_cache_sizes)
    
    combined = {}
    for cache_size in all_cache_sizes:
        total_miss = sum(get_miss_count_at_cache_size(pred, cache_size) 
                        for pred in predictions if pred)
        combined[cache_size] = total_miss
    
    return combined

def load_combined_deriche_predictions(base_dir_pattern):
    """Load and combine deriche1-6 predictions. Returns dict: cache_size -> miss_count."""
    predictions = []
    total_counts = []
    
    # Determine if this is FA or 12WA based on directory pattern
    is_fa = 'fully-associative' in base_dir_pattern or '-fa' in base_dir_pattern
    
    # Load predictions
    for i in range(1, 7):
        if is_fa:
            json_path = os.path.join('experiment/results', f'deriche{i}-fa', f'deriche{i}.json')
        else:
            json_path = os.path.join('experiment/results', f'deriche{i}-12way', f'deriche{i}.json')
        
        # For 12WA, we need FA total counts
        total_count_override = None
        if not is_fa:
            fa_json_path = os.path.join('experiment/results', f'deriche{i}-fa', f'deriche{i}.json')
            fa_pred = load_prediction_from_json_with_total(fa_json_path)
            if fa_pred:
                total_count_override = fa_pred['total_count']
        
        pred = load_prediction_from_json_with_total(json_path, total_count_override)
        if pred:
            predictions.append(pred)
            if is_fa:
                total_counts.append(pred['total_count'])
    
    combined = combine_miss_counts(predictions)
    
    # Calculate total access count (sum of all total counts)
    total_access = sum(p['total_count'] for p in predictions if p)
    
    return combined, total_access

def load_deriche_prediction(base_dir):
    """Load deriche prediction by combining deriche1-6."""
    combined, total_access = load_combined_deriche_predictions(base_dir)
    
    if not combined or total_access == 0:
        return None
    
    # Convert to miss ratio format
    turning_points = sorted(combined.keys())
    miss_ratios = [combined[cs] / total_access for cs in turning_points]
    
    return {
        'turning_points': turning_points,
        'miss_ratios': miss_ratios
    }


def compute_miss_ratios_for_program(program, sim_data_fa, sim_data_12wa, pred_data_fa, pred_data_12wa):
    """
    Compute miss ratios and relative errors for each cache size.
    Returns dict with:
    - 'FA-xK-SIM': simulated miss ratio for FA
    - 'FA-xK-PRED': predicted miss ratio for FA
    - 'FA-xK-ERROR': relative error = |sim_miss - pred_miss| / total_access
    - '12WA-xK-SIM': simulated miss ratio for 12WA
    - '12WA-xK-PRED': predicted miss ratio for 12WA
    - '12WA-xK-ERROR': relative error = |sim_miss - pred_miss| / total_access
    """
    ratios = {}
    for size_name, cache_size_bytes in CACHE_SIZES.items():
        # FA simulated
        sim_row_fa = sim_data_fa[sim_data_fa['d1_cache_size'] == cache_size_bytes]
        if sim_row_fa.empty:
            sim_ratio_fa = np.nan
            sim_miss_fa = np.nan
            total_access_fa = np.nan
        else:
            sim_miss_fa = sim_row_fa['d1_miss_count'].iloc[0]
            total_access_fa = sim_row_fa['total_access'].iloc[0]
            sim_ratio_fa = sim_miss_fa / total_access_fa if total_access_fa else np.nan
        
        # FA predicted
        pred_ratio_fa = get_prediction_miss_ratio(pred_data_fa, cache_size_bytes)
        
        # Calculate relative error for FA: |sim_miss - pred_miss| / total_access
        if np.isnan(sim_ratio_fa) or np.isnan(pred_ratio_fa) or np.isnan(total_access_fa):
            error_fa = np.nan
            pred_miss_fa = np.nan
        else:
            pred_miss_fa = pred_ratio_fa * total_access_fa
            error_fa = abs(sim_miss_fa - pred_miss_fa) / total_access_fa if total_access_fa else np.nan
        
        ratios[f'FA-{size_name}-SIM'] = sim_ratio_fa
        ratios[f'FA-{size_name}-PRED'] = pred_ratio_fa
        ratios[f'FA-{size_name}-ERROR'] = error_fa
        
        # 12WA simulated
        sim_row_12wa = sim_data_12wa[sim_data_12wa['d1_cache_size'] == cache_size_bytes]
        if sim_row_12wa.empty:
            sim_ratio_12wa = np.nan
            sim_miss_12wa = np.nan
            total_access_12wa = np.nan
        else:
            sim_miss_12wa = sim_row_12wa['d1_miss_count'].iloc[0]
            total_access_12wa = sim_row_12wa['total_access'].iloc[0]
            sim_ratio_12wa = sim_miss_12wa / total_access_12wa if total_access_12wa else np.nan
        
        # 12WA predicted
        pred_ratio_12wa = get_prediction_miss_ratio(pred_data_12wa, cache_size_bytes)
        
        # Calculate relative error for 12WA: |sim_miss - pred_miss| / total_access
        if np.isnan(sim_ratio_12wa) or np.isnan(pred_ratio_12wa) or np.isnan(total_access_12wa):
            error_12wa = np.nan
            pred_miss_12wa = np.nan
        else:
            pred_miss_12wa = pred_ratio_12wa * total_access_12wa
            error_12wa = abs(sim_miss_12wa - pred_miss_12wa) / total_access_12wa if total_access_12wa else np.nan
        
        ratios[f'12WA-{size_name}-SIM'] = sim_ratio_12wa
        ratios[f'12WA-{size_name}-PRED'] = pred_ratio_12wa
        ratios[f'12WA-{size_name}-ERROR'] = error_12wa
    
    # Compute statistics (mean, median) for each type
    for kind in ['FA-SIM', 'FA-PRED', '12WA-SIM', '12WA-PRED']:
        values = []
        for size_name in CACHE_SIZES.keys():
            if kind == 'FA-SIM':
                val = ratios.get(f'FA-{size_name}-SIM', np.nan)
            elif kind == 'FA-PRED':
                val = ratios.get(f'FA-{size_name}-PRED', np.nan)
            elif kind == '12WA-SIM':
                val = ratios.get(f'12WA-{size_name}-SIM', np.nan)
            else:  # 12WA-PRED
                val = ratios.get(f'12WA-{size_name}-PRED', np.nan)
            if not np.isnan(val):
                values.append(val)
        ratios[f'{kind}-MEAN'] = np.mean(values) if values else np.nan
        ratios[f'{kind}-MEDIAN'] = np.median(values) if values else np.nan
    
    # Compute average error for statistics columns
    for kind in ['FA', '12WA']:
        error_values = []
        for size_name in CACHE_SIZES.keys():
            err = ratios.get(f'{kind}-{size_name}-ERROR', np.nan)
            if not np.isnan(err):
                error_values.append(err)
        avg_error = np.mean(error_values) if error_values else np.nan
        ratios[f'{kind}-MEAN-ERROR'] = avg_error
        ratios[f'{kind}-MEDIAN-ERROR'] = avg_error
    
    return ratios

def main():
    parser = argparse.ArgumentParser(description='Generate comparison heatmap for polybench')
    parser.add_argument('--output', '-o', default='comparison_heatmap.svg',
                       help='Output file name (default: comparison_heatmap.svg)')
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
    programs_fa_sim = set(sim_fa['program'].unique())
    programs_12wa_sim = set(sim_12wa['program'].unique())
    programs_pred = set(total_counts.keys())
    programs = sorted(programs_fa_sim & programs_12wa_sim & programs_pred)
    print(f"Found {len(programs)} programs in common across all datasets")
    if len(programs) == 0:
        print("Error: No common programs found")
        sys.exit(1)
    
    # Columns: SIM and PRED for each cache size, plus statistics
    columns = []
    for size_name in CACHE_SIZES:
        columns += [f'FA-{size_name}-SIM', f'FA-{size_name}-PRED', 
                    f'12WA-{size_name}-SIM', f'12WA-{size_name}-PRED']
    # Add statistics columns
    for stat in ['MEAN', 'MEDIAN']:
        columns += [f'FA-SIM-{stat}', f'FA-PRED-{stat}', 
                    f'12WA-SIM-{stat}', f'12WA-PRED-{stat}']
    
    matrix_values = []  # For displaying miss ratios
    matrix_errors = []  # For coloring by relative error
    
    print("\nComputing miss ratios for each program...")
    for program in tqdm(programs, desc="Processing programs"):
        # Ad-hoc handling for deriche: combine deriche1-6 predictions
        if program == 'deriche':
            pred_data_fa = load_deriche_prediction(args.pred_fa_dir)
            pred_data_12wa = load_deriche_prediction(args.pred_12wa_dir)
        else:
            pred_fa_path = os.path.join(args.pred_fa_dir, f"{program}.json")
            pred_12wa_path = os.path.join(args.pred_12wa_dir, f"{program}.json")
            pred_data_fa = load_prediction_from_json(pred_fa_path)
            pred_data_12wa = load_prediction_from_json(pred_12wa_path)
        
        if pred_data_fa is None or pred_data_12wa is None:
            print(f"Warning: Missing prediction data for {program}, skipping")
            continue
        sim_data_fa = sim_fa[sim_fa['program'] == program]
        sim_data_12wa = sim_12wa[sim_12wa['program'] == program]
        if sim_data_fa.empty or sim_data_12wa.empty:
            print(f"Warning: Missing simulation data for {program}, skipping")
            continue
        ratios = compute_miss_ratios_for_program(
            program, sim_data_fa, sim_data_12wa,
            pred_data_fa, pred_data_12wa
        )
        
        # Build row for values (miss ratios) and errors
        row_values = [program]
        row_errors = [program]
        for col in columns:
            row_values.append(ratios.get(col, np.nan))
            # For error: SIM and PRED columns share the same error value
            if '-SIM' in col or '-PRED' in col:
                # Extract the error key by replacing -SIM/-PRED with -ERROR
                if '-SIM-' in col or '-PRED-' in col:
                    # Statistics columns like FA-SIM-MEAN or FA-PRED-MEAN
                    if '-SIM-' in col:
                        error_key = col.replace('-SIM-', '-') + '-ERROR'
                    else:
                        error_key = col.replace('-PRED-', '-') + '-ERROR'
                else:
                    # Regular columns like FA-3K-SIM or FA-3K-PRED
                    error_key = col.replace('-SIM', '-ERROR').replace('-PRED', '-ERROR')
                row_errors.append(ratios.get(error_key, np.nan))
            else:
                row_errors.append(np.nan)
        
        matrix_values.append(row_values)
        matrix_errors.append(row_errors)
    
    df_values = pd.DataFrame(matrix_values, columns=['Program'] + columns)
    df_values = df_values.set_index('Program')
    
    df_errors = pd.DataFrame(matrix_errors, columns=['Program'] + columns)
    df_errors = df_errors.set_index('Program')
    
    print(f"\nGenerated matrix with {len(df_values)} programs")
    
    # Determine max error from actual data
    error_max = df_errors.max().max()
    if np.isnan(error_max) or error_max <= 0:
        error_max = 0.2  # fallback
    print(f"Maximum error in data: {error_max:.4f}")
    
    print("Generating heatmap...")
    
    fig, ax = plt.subplots(figsize=(20, max(8, len(df_values) * 0.3)))
    
    # Create heatmap with values displayed but colored by errors
    # Use actual max error from data
    df_errors_normalized = df_errors.clip(upper=error_max)
    
    sns.heatmap(df_errors_normalized, annot=df_values, fmt='.4f', cmap='Wistia',
                cbar_kws={'label': f'Relative Error |sim_miss - pred_miss| / total_access (max: {error_max:.4f})'},
                linewidths=0.5, linecolor='white',
                vmin=0, vmax=error_max,
                ax=ax)
    
    ax.set_xlabel('Cache Configuration', fontsize=12)
    ax.set_ylabel('Program', fontsize=12)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    plt.tight_layout()
    plt.savefig(args.output, format='svg', bbox_inches='tight')
    print(f"\nHeatmap saved to: {args.output}")
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
        except Exception as e:
            print(f"Warning: Failed to convert to PDF: {e}")
            print("Make sure Inkscape is installed and in your PATH")
    print("\nDone!")
    # plt.show()

if __name__ == "__main__":
    main()
