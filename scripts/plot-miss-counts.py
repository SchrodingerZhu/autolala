#!/usr/bin/env python3
"""
Script to plot miss counts from multiple data sources based on configuration.
Creates a grid of subplots showing d1_miss_count vs d1_cache_size for each program.
Different associativities are plotted together in the same subplot.
Each row contains 4 plots maximum.

Supports both SQLite databases and JSON directory sources.
For JSON directories, loads miss_ratio_curve data and converts to miss counts.

Usage: python3 plot_miss_counts.py <plot_settings.json> [output_file]
       python3 plot_miss_counts.py scripts/plot_settings.json
       python3 plot_miss_counts.py scripts/plot_settings.json my_plot.png
"""

import sqlite3
import matplotlib.pyplot as plt
import pandas as pd
import sys
import math
import json
import os
from pathlib import Path
import numpy as np
from scipy import stats
from tabulate import tabulate

def load_settings(settings_path):
    """Load plot settings from JSON file."""
    with open(settings_path, 'r') as f:
        settings = json.load(f)
    
    # Validate input data entries
    supported_types = {'sqlite', 'json-directory'}
    for entry in settings['input-data']:
        if entry['type'] not in supported_types:
            raise NotImplementedError(f"Input type '{entry['type']}' is not yet implemented. Supported types: {supported_types}")
    
    return settings

def load_data_from_sqlite(db_path):
    """Load data from SQLite database."""
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
                # For now, extract the numeric part
                if ' ' in total_count_str:
                    # Format like "96000000 R" - extract the coefficient
                    total_count = int(total_count_str.split()[0])
                else:
                    total_count = int(total_count_str)
                
                total_counts[program_name] = total_count
                
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                print(f"Warning: Could not parse total count from {json_path}: {e}")
                
    return total_counts

def load_data_from_json_directory(json_dir_path, total_size_from_dir=None):
    """Load data from JSON directory containing miss ratio curves."""
    all_data = []
    
    if not os.path.exists(json_dir_path):
        print(f"Warning: JSON directory '{json_dir_path}' not found")
        return pd.DataFrame()
    
    # Load total counts if we need them from another directory
    total_counts = {}
    if total_size_from_dir:
        total_counts = load_total_counts_from_directory(total_size_from_dir)
    
    # Process each JSON file in the directory
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
            if total_size_from_dir and program_name in total_counts:
                # Use total count from base directory
                total_count = total_counts[program_name]
            else:
                # Try to get total count from current file
                total_count_str = data.get('total_count', '0')
                if ' ' in total_count_str:
                    # Format like "96000000 R" - extract the coefficient
                    total_count = int(total_count_str.split()[0])
                else:
                    total_count = int(total_count_str)
            
            if total_count == 0:
                print(f"Warning: Zero total count for {program_name}, skipping")
                continue
            
            # Convert data to miss counts
            for i, (turning_point, miss_ratio) in enumerate(zip(turning_points, miss_ratios)):
                # Convert turning point from blocks to bytes (multiply by 64 = 8x8)
                cache_size_bytes = turning_point * 64
                
                # Convert miss ratio to miss count
                miss_count = miss_ratio * total_count
                
                all_data.append({
                    'program': program_name,
                    'd1_cache_size': cache_size_bytes,
                    'd1_miss_count': miss_count,
                    'd1_associativity': 'from_json'  # Placeholder
                })
                
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"Warning: Could not parse {json_path}: {e}")
            continue
    
    if not all_data:
        return pd.DataFrame()
        
    return pd.DataFrame(all_data)

def load_all_data(settings):
    """Load data from all configured sources."""
    all_data = []
    
    for entry in settings['input-data']:
        if entry['type'] == 'sqlite':
            df = load_data_from_sqlite(entry['path'])
            if not df.empty:
                df['source_name'] = entry['name']
                df['linestyle'] = entry['linestyle']
                df['color'] = entry['color']
                df['alpha'] = entry.get('alpha', 0.8)  # Default alpha if not specified
                df['plot_style'] = 'line'  # Regular line plot for simulation data
                all_data.append(df)
                
        elif entry['type'] == 'json-directory':
            total_size_from = entry.get('total-size-from')
            df = load_data_from_json_directory(entry['path'], total_size_from)
            if not df.empty:
                df['source_name'] = entry['name']
                df['linestyle'] = entry['linestyle']
                df['color'] = entry['color']
                df['alpha'] = entry.get('alpha', 0.8)  # Default alpha if not specified
                df['plot_style'] = 'step'  # Step plot for turning point curves
                all_data.append(df)
                
        else:
            raise NotImplementedError(f"Input type '{entry['type']}' is not yet implemented.")
    
    # Combine all dataframes
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        return combined_df
    else:
        return pd.DataFrame()

def create_plots(df, output_path=None):
    """Create grid plots for all programs with different associativities."""
    programs = df['program'].unique()
    n_programs = len(programs)
    
    # Calculate grid dimensions (4 plots per row)
    n_cols = 4
    n_rows = math.ceil(n_programs / n_cols)
    
    # Create figure with subplots, leaving space for global legend
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    fig.suptitle('Cache Miss Counts vs Cache Size by Program (Different Associativities)', fontsize=16, y=0.98)
    
    # Track legend elements globally
    legend_elements = []
    legend_labels = []
    
    # Handle case where we have only one row
    if n_rows == 1:
        axes = axes.reshape(1, -1) if n_programs > 1 else [axes]
    
    # Flatten axes array for easier indexing
    axes_flat = axes.flatten() if n_programs > 1 else [axes]
    
    for i, program in enumerate(programs):
        ax = axes_flat[i]
        
        # Filter data for current program and remove zero miss counts
        program_data = df[df['program'] == program].copy()
        program_data = program_data[program_data['d1_miss_count'] > 0]
        
        if program_data.empty:
            # If no data after filtering zeros, show a message
            ax.text(0.5, 0.5, 'No non-zero\nmiss counts', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=10, alpha=0.7)
        else:
            # Group by source (different associativities)
            sources = program_data['source_name'].unique()
            
            for source in sources:
                source_data = program_data[program_data['source_name'] == source].copy()
                source_data = source_data.sort_values('d1_cache_size')
                
                if not source_data.empty:
                    # Get styling from the data
                    linestyle = source_data['linestyle'].iloc[0]
                    color = source_data['color'].iloc[0]
                    plot_style = source_data['plot_style'].iloc[0]
                    
                    # Get alpha value if available
                    alpha = source_data.get('alpha', pd.Series([0.8])).iloc[0]
                    
                    # Parse linestyle (e.g., "o-" means marker='o', linestyle='-')
                    if 'o' in linestyle:
                        marker = 'o'
                    elif 's' in linestyle:
                        marker = 's'
                    elif 'x' in linestyle or 'X' in linestyle:
                        marker = 'X'
                    else:
                        marker = None
                    
                    # Parse line style - check for dashed first, then solid
                    if '--' in linestyle:
                        line_style = '--'
                    elif '-' in linestyle:
                        line_style = '-'
                    else:
                        line_style = 'None'
                    
                    # Plot based on style
                    if plot_style == 'step':
                        # Use step plot for turning point curves
                        line, = ax.step(source_data['d1_cache_size'], source_data['d1_miss_count'], 
                                       where='post', marker=marker, linestyle=line_style, 
                                       linewidth=2, markersize=4, color=color, alpha=alpha, label=source)
                    else:
                        # For simulation data: use scatter with X markers if it's set-associative (not fully associative)
                        # Identify set-associative by checking for keywords in the source name
                        is_set_associative = 'way associative' in source.lower() and 'fully' not in source.lower()
                        
                        if is_set_associative:
                            # Scatter plot with X markers for set-associative simulation
                            line = ax.scatter(source_data['d1_cache_size'], source_data['d1_miss_count'],
                                            marker='X', s=100, color=color, alpha=alpha, label=source, zorder=3)
                        else:
                            # Regular line plot for fully associative simulation data
                            line, = ax.plot(source_data['d1_cache_size'], source_data['d1_miss_count'], 
                                           marker=marker, linestyle=line_style, linewidth=2, markersize=4,
                                           color=color, alpha=alpha, label=source)
                    
                    # Add to global legend (avoid duplicates)
                    if source not in legend_labels:
                        legend_elements.append(line)
                        legend_labels.append(source)
                        
                else:
                    # Handle empty curves with error annotation
                    ax.annotate(f'{source}: Empty curve', xy=(0.5, 0.1), 
                               xycoords='axes fraction', fontsize=8, color='red', alpha=0.7)
        
        # Set labels and title
        ax.set_xlabel('Cache Size (bytes)')
        ax.set_ylabel('Miss Count')
        ax.set_title(f'{program}', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Use log scale for x-axis if cache sizes span multiple orders of magnitude
        if not program_data.empty:
            cache_sizes = program_data['d1_cache_size'].unique()
            cache_sizes = cache_sizes[cache_sizes > 0]  # Filter out zero values
            if len(cache_sizes) > 1 and min(cache_sizes) > 0 and max(cache_sizes) / min(cache_sizes) > 10:
                ax.set_xscale('log')
        
        # Format y-axis for readability
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # Hide unused subplots
    for i in range(n_programs, len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    # Add global legend
    if legend_elements:
        fig.legend(legend_elements, legend_labels, 
                  loc='upper center', bbox_to_anchor=(0.5, 0.96), 
                  ncol=min(len(legend_labels), 4), fontsize=12)
    
    # Adjust layout to make room for legend
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  # Make room for the global legend
    
    # Save or show plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()
    
    return fig

def get_total_accesses(base_data, total_size_from_dir=None, program_name=None):
    """Get total accesses for calculating relative error.
    
    Priority order:
    1. If total_size_from_dir is specified, load total_count from that directory
    2. Otherwise, estimate from base_data (miss count at smallest cache size)
    """
    if total_size_from_dir and program_name:
        try:
            json_path = os.path.join(total_size_from_dir, f"{program_name}.json")
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    total_count_str = data.get('total_count', '0')
                    
                    # Parse the total count (may contain 'R' for symbolic values)
                    if ' ' in total_count_str:
                        # Format like "96000000 R" - extract the coefficient
                        total_count = int(total_count_str.split()[0])
                    else:
                        total_count = int(total_count_str)
                    
                    if total_count > 0:
                        return total_count
        except (json.JSONDecodeError, ValueError, FileNotFoundError, KeyError):
            pass  # Fall back to estimation
    
    # Fallback: estimate from base data
    if base_data.empty:
        return 1  # Default to avoid division by zero
    
    # Sort by cache size and get the miss count at smallest cache size
    sorted_data = base_data.sort_values('d1_cache_size')
    max_miss_count = sorted_data['d1_miss_count'].iloc[0]  # Smallest cache = most misses
    
    # Use the maximum miss count as an approximation of total accesses
    # This is reasonable since at very small cache sizes, miss rate approaches 1
    return max(max_miss_count, 1)  # Ensure non-zero

def calculate_metrics(base_data, compare_data, total_size_from_dir=None, program_name=None):
    """Calculate relative error between base and compare datasets.
    
    Uses new alignment strategy:
    1. Find minimum of maximum cache sizes from both datasets
    2. Step through cache sizes using powers of 2 for reasonable coverage
    3. Interpolate both datasets at each step
    4. Calculate relative error = |base - compare| / total_accesses
    """
    if base_data.empty or compare_data.empty:
        return {'relative_error': np.nan, 'num_points': 0}
    
    # Sort both datasets by cache size
    base_sorted = base_data.sort_values('d1_cache_size').copy()
    compare_sorted = compare_data.sort_values('d1_cache_size').copy()
    
    # Find the minimum of maximum cache sizes
    base_max = base_sorted['d1_cache_size'].max()
    compare_max = compare_sorted['d1_cache_size'].max()
    max_cache_size = min(base_max, compare_max)
    
    if max_cache_size < 1:
        return {'relative_error': np.nan, 'num_points': 0}
    
    # Get total accesses (try from total_size_from_dir first, then estimate)
    total_accesses = get_total_accesses(base_sorted, total_size_from_dir, program_name)
    
    # Use base dataset's cache sizes as comparison points (only those within compare range)
    base_cache_sizes = base_sorted['d1_cache_size'].tolist()
    compare_min = compare_sorted['d1_cache_size'].min()
    compare_max = compare_sorted['d1_cache_size'].max()
    
    # Filter base cache sizes to only those within the compare dataset's range
    # to ensure we can interpolate accurately
    cache_sizes = [size for size in base_cache_sizes if compare_min <= size <= compare_max]
    
    if len(cache_sizes) < 2:
        return {'relative_error': np.nan, 'num_points': 0}
    
    # Get base values for the filtered cache sizes (no interpolation needed)
    base_data_filtered = base_sorted[base_sorted['d1_cache_size'].isin(cache_sizes)].sort_values('d1_cache_size')
    
    # Interpolate compare dataset to match the filtered base cache sizes  
    compare_interp = interpolate_data(compare_sorted, cache_sizes)
    
    if base_data_filtered.empty or compare_interp.empty:
        return {'relative_error': np.nan, 'num_points': 0}
    
    # Create mappings for easier lookup
    base_dict = dict(zip(base_data_filtered['d1_cache_size'], base_data_filtered['d1_miss_count']))
    compare_dict = dict(zip(compare_interp['d1_cache_size'], compare_interp['d1_miss_count']))
    
    # Calculate relative errors for common cache sizes
    relative_errors = []
    
    for cache_size in cache_sizes:
        if cache_size in base_dict and cache_size in compare_dict:
            base_val = base_dict[cache_size]
            compare_val = compare_dict[cache_size]
            
            # Calculate relative error = |base - compare| / total_accesses
            if np.isfinite(base_val) and np.isfinite(compare_val):
                rel_error = abs(base_val - compare_val) / total_accesses
                relative_errors.append(rel_error)
    
    if not relative_errors:
        return {'relative_error': np.nan, 'num_points': 0}
    
    # Calculate mean relative error
    mean_relative_error = np.mean(relative_errors)
    
    return {'relative_error': mean_relative_error, 'num_points': len(relative_errors)}

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
            # No data available, skip
            continue
        elif smaller.empty:
            # Use the smallest available value (extrapolate)
            miss_count = larger['d1_miss_count'].iloc[0]
        elif larger.empty:
            # Use the largest available value (extrapolate)
            miss_count = smaller['d1_miss_count'].iloc[-1]
        elif cache_size in df_sorted['d1_cache_size'].values:
            # Exact match
            miss_count = df_sorted[df_sorted['d1_cache_size'] == cache_size]['d1_miss_count'].iloc[0]
        else:
            # Interpolation between two points
            x1, y1 = smaller['d1_cache_size'].iloc[-1], smaller['d1_miss_count'].iloc[-1]
            x2, y2 = larger['d1_cache_size'].iloc[0], larger['d1_miss_count'].iloc[0]
            
            # Handle edge cases
            if x1 == x2:
                miss_count = (y1 + y2) / 2
            elif y1 > 0 and y2 > 0 and x1 > 0 and x2 > 0:
                # Logarithmic interpolation (works well for cache miss curves)
                try:
                    log_y1, log_y2 = np.log(y1), np.log(y2)
                    log_x1, log_x2 = np.log(x1), np.log(x2)
                    log_cache_size = np.log(cache_size)
                    
                    log_miss_count = log_y1 + (log_y2 - log_y1) * (log_cache_size - log_x1) / (log_x2 - log_x1)
                    miss_count = np.exp(log_miss_count)
                except (ValueError, OverflowError):
                    # Fallback to linear interpolation
                    miss_count = y1 + (y2 - y1) * (cache_size - x1) / (x2 - x1)
            else:
                # Linear interpolation fallback
                miss_count = y1 + (y2 - y1) * (cache_size - x1) / (x2 - x1)
        
        # Ensure the result is valid
        if np.isfinite(miss_count):
            interpolated_data.append({
                'd1_cache_size': cache_size,
                'd1_miss_count': max(0, miss_count)  # Ensure non-negative
            })
    
    return pd.DataFrame(interpolated_data)

def process_comparisons(df, comparisons):
    """Process all comparison configurations and calculate metrics."""
    results = []
    
    for comparison in comparisons:
        base_name = comparison['base']
        compare_name = comparison['compare']
        requested_metrics = comparison.get('metric', ['relative_error'])  # Default to relative_error
        summarize_methods = comparison.get('summarize', ['geomean', 'mean'])
        total_size_from_dir = comparison.get('total-size-from')  # Get total size directory
        
        # Get data for base and compare sources
        base_df = df[df['source_name'] == base_name]
        compare_df = df[df['source_name'] == compare_name]
        
        if base_df.empty or compare_df.empty:
            print(f"Warning: Missing data for comparison {base_name} vs {compare_name}")
            continue
        
        # Calculate metrics for each program
        programs = set(base_df['program'].unique()).intersection(set(compare_df['program'].unique()))
        
        program_metrics = {}
        for program in programs:
            base_program = base_df[base_df['program'] == program]
            compare_program = compare_df[compare_df['program'] == program]
            
            metrics = calculate_metrics(base_program, compare_program, total_size_from_dir, program)
            program_metrics[program] = metrics
        
        # Summarize relative error across programs
        metric_values = [program_metrics[prog]['relative_error'] for prog in program_metrics 
                        if not np.isnan(program_metrics[prog]['relative_error'])]
        
        if metric_values:
            for summary_method in summarize_methods:
                if summary_method == 'mean':
                    summary_value = np.mean(metric_values)
                elif summary_method == 'geomean':
                    # Geometric mean (use only positive values)
                    positive_values = [v for v in metric_values if v > 0]
                    if positive_values:
                        summary_value = stats.gmean(positive_values)
                    else:
                        summary_value = np.nan
                else:
                    summary_value = np.nan
                
                results.append({
                    'comparison': f"{base_name} vs {compare_name}",
                    'metric': 'RELATIVE_ERROR',
                    'summary_method': summary_method,
                    'value': summary_value,
                    'num_programs': len(metric_values),
                    'program_details': program_metrics
                })
    
    return results

def print_comparison_results(results):
    """Print comparison results in fancy unicode tables."""
    if not results:
        print("No comparison results to display.")
        return
    
    # Print summary table
    print("\n" + "="*80)
    print("📊 COMPARISON METRICS SUMMARY")
    print("="*80)
    
    summary_data = []
    for result in results:
        summary_data.append([
            result['comparison'],
            result['metric'],
            result['summary_method'].title(),
            f"{result['value']:.6e}" if not np.isnan(result['value']) else "N/A",
            result['num_programs']
        ])
    
    headers = ['Comparison', 'Metric', 'Summary', 'Value', '# Programs']
    print(tabulate(summary_data, headers=headers, tablefmt='fancy_grid', floatfmt='.6e'))
    
    # Print detailed per-program results
    print("\n" + "="*80)
    print("📋 DETAILED PER-PROGRAM METRICS")
    print("="*80)
    
    # Group results by comparison
    comparison_groups = {}
    for result in results:
        comp_key = result['comparison']
        if comp_key not in comparison_groups:
            comparison_groups[comp_key] = result['program_details']
    
    for comparison, program_details in comparison_groups.items():
        print(f"\n🔍 {comparison}")
        print("-" * len(f"🔍 {comparison}"))
        
        detailed_data = []
        for program, metrics in program_details.items():
            rel_error_val = f"{metrics['relative_error']:.6e}" if not np.isnan(metrics['relative_error']) else "N/A"
            detailed_data.append([
                program,
                rel_error_val,
                metrics['num_points']
            ])
        
        detailed_headers = ['Program', 'Relative Error', 'Points']
        print(tabulate(detailed_data, headers=detailed_headers, tablefmt='fancy_grid'))

def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3 or sys.argv[1] in ['-h', '--help']:
        print("Usage: python3 plot_miss_counts.py <plot_settings.json> [output_file]")
        print()
        print("Arguments:")
        print("  plot_settings.json   JSON configuration file with input data sources")
        print("  output_file         Optional output PNG file name (default: miss_counts_comparison.png)")
        print()
        print("Examples:")
        print("  python3 plot_miss_counts.py scripts/plot_settings.json")
        print("  python3 plot_miss_counts.py scripts/plot_settings.json my_plot.png")
        print()
        print("The script creates a grid plot showing d1_miss_count vs d1_cache_size")
        print("for each program, comparing different associativities on the same plot.")
        print("Supports both SQLite databases and JSON directories with miss ratio curves.")
        sys.exit(0 if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help'] else 1)
    
    settings_path = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) == 3 else None
    
    # Check if settings file exists
    if not Path(settings_path).exists():
        print(f"Error: Settings file '{settings_path}' not found.")
        sys.exit(1)
    
    try:
        # Load settings
        print(f"Loading settings from {settings_path}...")
        settings = load_settings(settings_path)
        
        # Load all data sources
        print("Loading data from configured sources...")
        df = load_all_data(settings)
        
        if df.empty:
            print("Error: No data found in any configured source.")
            sys.exit(1)
        
        programs = df['program'].unique()
        sources = df['source_name'].unique()
        print(f"Found {len(programs)} programs across {len(sources)} data sources")
        print("Sources:", ", ".join(sources))
        print("Programs:", ", ".join(programs[:5]) + ("..." if len(programs) > 5 else ""))
        
        # Generate output filename
        if output_file:
            output_path = output_file
        else:
            output_path = "miss_counts_comparison.png"
        
        # Create plots
        print("Creating comparison plots...")
        fig = create_plots(df, output_path)
        
        # Process comparisons if specified in settings
        if 'comparison' in settings and settings['comparison']:
            print("\nProcessing comparison metrics...")
            comparison_results = process_comparisons(df, settings['comparison'])
            print_comparison_results(comparison_results)
        
        print("Done!")
        
    except NotImplementedError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"JSON parsing error in settings file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
