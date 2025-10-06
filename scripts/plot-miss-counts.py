#!/usr/bin/env python3
"""
Script to plot miss counts from multiple SQLite databases based on configuration.
Creates a grid of subplots showing d1_miss_count vs d1_cache_size for each program.
Different associativities are plotted together in the same subplot.
Each row contains 4 plots maximum.

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
from pathlib import Path

def load_settings(settings_path):
    """Load plot settings from JSON file."""
    with open(settings_path, 'r') as f:
        settings = json.load(f)
    
    # Validate input data entries
    for entry in settings['input-data']:
        if entry['type'] != 'sqlite':
            raise NotImplementedError(f"Input type '{entry['type']}' is not yet implemented. Only 'sqlite' is supported.")
    
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
    return df

def load_all_data(settings):
    """Load data from all configured sources."""
    all_data = []
    
    for entry in settings['input-data']:
        if entry['type'] == 'sqlite':
            df = load_data_from_sqlite(entry['path'])
            df['source_name'] = entry['name']
            df['linestyle'] = entry['linestyle']
            df['color'] = entry['color']
            all_data.append(df)
        else:
            raise NotImplementedError(f"Input type '{entry['type']}' is not yet implemented.")
    
    # Combine all dataframes
    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df

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
                    
                    # Parse linestyle (e.g., "o-" means marker='o', linestyle='-')
                    marker = 'o' if 'o' in linestyle else None
                    line_style = '-' if '-' in linestyle else 'None'
                    
                    line, = ax.plot(source_data['d1_cache_size'], source_data['d1_miss_count'], 
                                   marker=marker, linestyle=line_style, linewidth=2, markersize=4,
                                   color=color, label=source)
                    
                    # Add to global legend (avoid duplicates)
                    if source not in legend_labels:
                        legend_elements.append(line)
                        legend_labels.append(source)
        
        # Set labels and title
        ax.set_xlabel('Cache Size (bytes)')
        ax.set_ylabel('Miss Count')
        ax.set_title(f'{program}', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Use log scale for x-axis if cache sizes span multiple orders of magnitude
        if not program_data.empty:
            cache_sizes = program_data['d1_cache_size'].unique()
            if len(cache_sizes) > 1 and max(cache_sizes) / min(cache_sizes) > 10:
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
        sys.exit(1)

if __name__ == "__main__":
    main()
