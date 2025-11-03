#!/usr/bin/env python3
"""
Script to find the optimal skewness value by testing different skewness values
from 1.2 to 1.8 with 0.01 precision and comparing mean errors.

This script:
1. Generates configurations for each skewness value using assoc-conv
2. Compares the analytical results with simulation data from SQLite database
3. Calculates mean relative error for each program and overall
4. Finds the skewness value that minimizes the mean error
"""

import os
import sys
import json
import subprocess
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats
from tabulate import tabulate

def load_excluded_programs(settings_path="scripts/plot-settings-einsum.json"):
    """Load excluded programs from plot settings."""
    try:
        with open(settings_path, 'r') as f:
            settings = json.load(f)
        return settings.get('excluded_programs', [])
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"Warning: Could not load excluded programs from {settings_path}")
        return []

def run_conversion_for_skewness(skewness):
    """Run the conversion process for a specific skewness value."""
    skewness_str = f"{skewness:.2f}"
    output_dir = f"results/12-way-einsum-{skewness_str}"
    
    print(f"Processing skewness {skewness_str}...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of input files
    input_dir = "results/fully-associative-einsum"
    if not os.path.exists(input_dir):
        print(f"Error: Input directory {input_dir} not found")
        return False
    
    success_count = 0
    total_count = 0
    
    for json_file in os.listdir(input_dir):
        if not json_file.endswith('.json'):
            continue
            
        program_name = json_file.replace('.json', '')
        input_path = os.path.join(input_dir, json_file)
        output_path = os.path.join(output_dir, json_file)
        
        total_count += 1
        
        # Run cargo command
        cmd = [
            "cargo", "run", "--release", "--quiet", "--bin", "assoc-conv",
            "--", "-o", output_path, "-i", input_path,
            "-a", "12", "-s", str(skewness), "-d", "constant"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                success_count += 1
                print(f"  ✓ {program_name}")
            else:
                print(f"  ✗ {program_name}: {result.stderr.strip()}")
        except subprocess.TimeoutExpired:
            print(f"  ✗ {program_name}: Timeout")
        except Exception as e:
            print(f"  ✗ {program_name}: {e}")
    
    print(f"Completed {success_count}/{total_count} conversions for skewness {skewness_str}")
    return success_count > 0

def load_simulation_data(db_path="results/12-way-einsum.db"):
    """Load simulation data from SQLite database."""
    if not os.path.exists(db_path):
        print(f"Error: Database {db_path} not found")
        return pd.DataFrame()
    
    conn = sqlite3.connect(db_path)
    query = """
    SELECT program, d1_cache_size, d1_miss_count, d1_associativity 
    FROM records 
    ORDER BY program, d1_cache_size
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Remove .mlir suffix from program names to match JSON files
    df['program'] = df['program'].str.replace('.mlir', '', regex=False)
    
    return df

def load_total_counts_from_directory(base_dir="results/fully-associative-einsum"):
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

def load_analytical_data(json_dir_path, total_counts):
    """Load analytical data from JSON directory."""
    all_data = []
    
    if not os.path.exists(json_dir_path):
        print(f"Warning: JSON directory '{json_dir_path}' not found")
        return pd.DataFrame()
    
    for json_file in os.listdir(json_dir_path):
        if not json_file.endswith('.json'):
            continue
            
        program_name = json_file.replace('.json', '')
        json_path = os.path.join(json_dir_path, json_file)
        
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            miss_ratio_curve = data.get('miss_ratio_curve', {})
            
            if not miss_ratio_curve:
                print(f"Warning: No miss_ratio_curve found in {json_file}")
                continue
                
            turning_points = miss_ratio_curve.get('turning_points', [])
            miss_ratios = miss_ratio_curve.get('miss_ratio', [])
            
            if not turning_points or not miss_ratios:
                print(f"Warning: Empty miss ratio curve in {json_file}")
                continue
            
            # Get total count
            total_count = total_counts.get(program_name, 0)
            
            if total_count == 0:
                print(f"Warning: Zero total count for {program_name}, skipping")
                continue
            
            # Convert data to miss counts
            for turning_point, miss_ratio in zip(turning_points, miss_ratios):
                # Convert turning point from blocks to bytes (multiply by 64 = 8x8)
                cache_size_bytes = turning_point * 64
                
                # Convert miss ratio to miss count
                miss_count = miss_ratio * total_count
                
                all_data.append({
                    'program': program_name,
                    'd1_cache_size': cache_size_bytes,
                    'd1_miss_count': miss_count
                })
                
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"Warning: Could not parse {json_path}: {e}")
            continue
    
    if not all_data:
        return pd.DataFrame()
        
    return pd.DataFrame(all_data)

def interpolate_data(df, cache_sizes):
    """Interpolate miss counts for common cache sizes."""
    if df.empty or not cache_sizes:
        return pd.DataFrame(columns=['d1_cache_size', 'd1_miss_count'])
    
    # Sort the input data by cache size
    df_sorted = df.sort_values('d1_cache_size').copy()
    
    interpolated_data = []
    
    for cache_size in cache_sizes:
        # Find the closest cache size points for interpolation
        smaller = df_sorted[df_sorted['d1_cache_size'] <= cache_size]
        larger = df_sorted[df_sorted['d1_cache_size'] >= cache_size]
        
        if smaller.empty and larger.empty:
            continue
        elif smaller.empty:
            miss_count = larger['d1_miss_count'].iloc[0]
        elif larger.empty:
            miss_count = smaller['d1_miss_count'].iloc[-1]
        elif cache_size in df_sorted['d1_cache_size'].values:
            miss_count = df_sorted[df_sorted['d1_cache_size'] == cache_size]['d1_miss_count'].iloc[0]
        else:
            # Linear interpolation
            x1, y1 = smaller['d1_cache_size'].iloc[-1], smaller['d1_miss_count'].iloc[-1]
            x2, y2 = larger['d1_cache_size'].iloc[0], larger['d1_miss_count'].iloc[0]
            
            if x1 == x2:
                miss_count = (y1 + y2) / 2
            else:
                miss_count = y1 + (y2 - y1) * (cache_size - x1) / (x2 - x1)
        
        # Ensure the result is valid
        if np.isfinite(miss_count):
            interpolated_data.append({
                'd1_cache_size': cache_size,
                'd1_miss_count': max(0, miss_count)
            })
    
    return pd.DataFrame(interpolated_data)

def calculate_error_for_program(sim_data, analytical_data, total_accesses):
    """Calculate mean relative error for a single program."""
    if sim_data.empty or analytical_data.empty:
        return np.nan
    
    # Sort both datasets by cache size
    sim_sorted = sim_data.sort_values('d1_cache_size').copy()
    analytical_sorted = analytical_data.sort_values('d1_cache_size').copy()
    
    # Find common cache size range
    sim_min, sim_max = sim_sorted['d1_cache_size'].min(), sim_sorted['d1_cache_size'].max()
    analytical_min, analytical_max = analytical_sorted['d1_cache_size'].min(), analytical_sorted['d1_cache_size'].max()
    
    # Use simulation cache sizes as reference points within analytical range
    sim_cache_sizes = sim_sorted['d1_cache_size'].tolist()
    common_cache_sizes = [size for size in sim_cache_sizes 
                         if analytical_min <= size <= analytical_max]
    
    if len(common_cache_sizes) < 2:
        return np.nan
    
    # Get simulation values (no interpolation needed)
    sim_filtered = sim_sorted[sim_sorted['d1_cache_size'].isin(common_cache_sizes)]
    
    # Interpolate analytical data to match simulation cache sizes
    analytical_interp = interpolate_data(analytical_sorted, common_cache_sizes)
    
    if sim_filtered.empty or analytical_interp.empty:
        return np.nan
    
    # Create mappings for easier lookup
    sim_dict = dict(zip(sim_filtered['d1_cache_size'], sim_filtered['d1_miss_count']))
    analytical_dict = dict(zip(analytical_interp['d1_cache_size'], analytical_interp['d1_miss_count']))
    
    # Calculate relative errors
    relative_errors = []
    
    for cache_size in common_cache_sizes:
        if cache_size in sim_dict and cache_size in analytical_dict:
            sim_val = sim_dict[cache_size]
            analytical_val = analytical_dict[cache_size]
            
            # Calculate relative error = |difference| / actual (simulation is actual)
            if np.isfinite(sim_val) and np.isfinite(analytical_val) and sim_val > 0:
                rel_error = abs(sim_val - analytical_val) / sim_val
                relative_errors.append(rel_error)
    
    if not relative_errors:
        return np.nan
    
    return np.mean(relative_errors)

def evaluate_skewness(skewness, sim_data, total_counts, excluded_programs):
    """Evaluate the error for a specific skewness value."""
    skewness_str = f"{skewness:.2f}"
    analytical_dir = f"results/12-way-einsum-{skewness_str}"
    
    if not os.path.exists(analytical_dir):
        print(f"Warning: Directory {analytical_dir} not found")
        return np.nan, {}
    
    # Load analytical data
    analytical_data = load_analytical_data(analytical_dir, total_counts)
    
    if analytical_data.empty:
        print(f"Warning: No analytical data found for skewness {skewness_str}")
        return np.nan, {}
    
    # Get common programs (excluding problematic ones)
    sim_programs = set(sim_data['program'].unique())
    analytical_programs = set(analytical_data['program'].unique())
    common_programs = sim_programs.intersection(analytical_programs)
    
    # Remove excluded programs
    common_programs = common_programs - set(excluded_programs)
    
    program_errors = {}
    
    for program in common_programs:
        sim_program = sim_data[sim_data['program'] == program]
        analytical_program = analytical_data[analytical_data['program'] == program]
        
        total_accesses = total_counts.get(program, 1)  # Fallback to 1 to avoid division by zero
        
        error = calculate_error_for_program(sim_program, analytical_program, total_accesses)
        program_errors[program] = error
    
    # Calculate mean error across all programs
    valid_errors = [err for err in program_errors.values() if not np.isnan(err)]
    
    if not valid_errors:
        return np.nan, program_errors
    
    mean_error = np.mean(valid_errors)
    return mean_error, program_errors

def main():
    print("🔍 Skewness Optimization for Einsum Cache Analysis")
    print("=" * 60)
    
    # Load excluded programs
    excluded_programs = load_excluded_programs()
    print(f"Excluded programs: {excluded_programs}")
    
    # Load simulation data
    print("\n📊 Loading simulation data...")
    sim_data = load_simulation_data()
    if sim_data.empty:
        print("Error: Could not load simulation data")
        return 1
    
    # Load total counts
    print("📊 Loading total counts...")
    total_counts = load_total_counts_from_directory()
    if not total_counts:
        print("Error: Could not load total counts")
        return 1
    
    # Define skewness range
    skewness_values = np.arange(1.2, 1.81, 0.01)  # 1.2 to 1.8 with 0.01 precision
    
    print(f"\n🔧 Testing {len(skewness_values)} skewness values from 1.2 to 1.8...")
    
    results = []
    detailed_results = {}
    
    for i, skewness in enumerate(skewness_values):
        print(f"\n[{i+1}/{len(skewness_values)}] Testing skewness {skewness:.2f}")
        
        # Run conversion for this skewness
        success = run_conversion_for_skewness(skewness)
        if not success:
            print(f"  ⚠️  Conversion failed for skewness {skewness:.2f}")
            continue
        
        # Evaluate the error
        mean_error, program_errors = evaluate_skewness(skewness, sim_data, total_counts, excluded_programs)
        
        if not np.isnan(mean_error):
            results.append((skewness, mean_error))
            detailed_results[skewness] = program_errors
            print(f"  📈 Mean error: {mean_error:.6f}")
        else:
            print(f"  ⚠️  Could not calculate error for skewness {skewness:.2f}")
    
    if not results:
        print("\n❌ No valid results obtained")
        return 1
    
    # Find optimal skewness
    results.sort(key=lambda x: x[1])  # Sort by error
    optimal_skewness, optimal_error = results[0]
    
    print(f"\n🎯 RESULTS")
    print("=" * 60)
    print(f"Optimal skewness: {optimal_skewness:.2f}")
    print(f"Minimum mean error: {optimal_error:.6f}")
    
    # Print top 10 best results
    print(f"\n📊 Top 10 Best Skewness Values:")
    print("-" * 40)
    for i, (skewness, error) in enumerate(results[:10]):
        print(f"{i+1:2d}. Skewness {skewness:.2f}: {error:.6f}")
    
    # Save detailed results
    output_file = "data/results/skewness_optimization_results.txt"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write("Skewness Optimization Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Optimal skewness: {optimal_skewness:.2f}\n")
        f.write(f"Minimum mean error: {optimal_error:.6f}\n\n")
        
        f.write("All Results (sorted by error):\n")
        f.write("-" * 30 + "\n")
        for skewness, error in results:
            f.write(f"Skewness {skewness:.2f}: {error:.6f}\n")
        
        f.write(f"\nDetailed Program Errors for Optimal Skewness ({optimal_skewness:.2f}):\n")
        f.write("-" * 50 + "\n")
        if optimal_skewness in detailed_results:
            for program, error in detailed_results[optimal_skewness].items():
                if not np.isnan(error):
                    f.write(f"{program}: {error:.6f}\n")
    
    print(f"\n💾 Detailed results saved to {output_file}")
    
    # Create a simple plot
    if len(results) > 1:
        print("\n📈 Creating optimization plot...")
        
        skewness_vals = [r[0] for r in results]
        error_vals = [r[1] for r in results]
        
        plt.figure(figsize=(12, 8))
        plt.plot(skewness_vals, error_vals, 'b-o', markersize=4, linewidth=1)
        plt.axvline(x=optimal_skewness, color='red', linestyle='--', 
                   label=f'Optimal: {optimal_skewness:.2f}')
        plt.xlabel('Skewness Value')
        plt.ylabel('Mean Relative Error')
        plt.title('Cache Miss Prediction Error vs Skewness Value')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plot_file = "data/graphs/skewness_optimization.png"
        os.makedirs(os.path.dirname(plot_file), exist_ok=True)
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"📊 Plot saved to {plot_file}")
        plt.close()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
