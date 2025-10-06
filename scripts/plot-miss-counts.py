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
                        # Regular line plot for simulation data
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
