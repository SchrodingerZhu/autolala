#!/usr/bin/env python3
"""
Simple script to plot D1 miss count curves from SQLite database.
"""

import sqlite3
import matplotlib.pyplot as plt
import pandas as pd

def main():
    # Database connection
    db_path = '/home/schrodingerzy/Documents/contractions/data-fully-associative-matrix.db'
    conn = sqlite3.connect(db_path)
    
    # Read data for both programs
    query = """
    SELECT program, d1_associativity as cache_size, d1_miss_count 
    FROM records 
    WHERE program IN ('tiled_batched_gemm.mlir', 'tiled_matrix_matrix.mlir')
    ORDER BY program, d1_associativity
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot each program
    for program in df['program'].unique():
        program_data = df[df['program'] == program]
        clean_name = program.replace('.mlir', '').replace('_', ' ').title()
        plt.plot(program_data['cache_size'], 
                program_data['d1_miss_count'], 
                marker='o', 
                linewidth=2, 
                markersize=6,
                label=clean_name,
                drawstyle='steps-post')
    
    plt.xlabel('Cache Size (blocks)')
    plt.ylabel('D1 Miss Count')
    plt.title('D1 Miss Count vs Cache Size (Fully Associative)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Use log scale if needed for better visualization
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig('d1_miss_curves_simple.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Plot saved as 'd1_miss_curves_simple.png'")
    print("\nData summary:")
    for program in df['program'].unique():
        program_data = df[df['program'] == program]
        print(f"{program}: {len(program_data)} data points")
        print(f"  Cache size range: {program_data['cache_size'].min()} - {program_data['cache_size'].max()}")
        print(f"  Miss count range: {program_data['d1_miss_count'].min():,} - {program_data['d1_miss_count'].max():,}")

if __name__ == "__main__":
    main()