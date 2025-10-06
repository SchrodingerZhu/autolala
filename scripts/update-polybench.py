#!/usr/bin/env python3
"""
PolyBench Configuration Updater

This script automatically updates PolyBench .c files with medium dataset sizes
from the upstream PolyBenchC-4.2.1 repository. It:

1. Clones the upstream repository to a temporary directory
2. Extracts medium dataset sizes from header files
3. Updates corresponding .c files with the new sizes
4. Adds GitHub URL comments indicating the source configuration
5. Cleans up temporary files

Usage:
    python3 update_polybench.py

The script is fully portable and handles its own dependencies by cloning
the upstream repository as needed.

Author: GitHub Copilot
Date: October 2025
"""

import os
import re
import glob
import subprocess
import tempfile
import shutil
from pathlib import Path

# Mapping of file names to their header file paths in the upstream repository
HEADER_FILES = {
    "2mm": "linear-algebra/kernels/2mm/2mm.h",
    "3mm": "linear-algebra/kernels/3mm/3mm.h",
    "adi": "stencils/adi/adi.h",
    "atax": "linear-algebra/kernels/atax/atax.h",
    "bicg": "linear-algebra/kernels/bicg/bicg.h",
    "cholesky": "linear-algebra/solvers/cholesky/cholesky.h",
    "correlation": "datamining/correlation/correlation.h",
    "covariance": "datamining/covariance/covariance.h",
    "deriche": "medley/deriche/deriche.h",
    "doitgen": "linear-algebra/kernels/doitgen/doitgen.h",
    "durbin": "linear-algebra/solvers/durbin/durbin.h",
    "fdtd-2d": "stencils/fdtd-2d/fdtd-2d.h",
    "gemm": "linear-algebra/blas/gemm/gemm.h",
    "gemver": "linear-algebra/blas/gemver/gemver.h",
    "gesummv": "linear-algebra/blas/gesummv/gesummv.h",
    "gramschmidt": "linear-algebra/solvers/gramschmidt/gramschmidt.h",
    "heat-3d": "stencils/heat-3d/heat-3d.h",
    "jacobi-1d": "stencils/jacobi-1d/jacobi-1d.h",
    "jacobi-2d": "stencils/jacobi-2d/jacobi-2d.h",
    "lu": "linear-algebra/solvers/lu/lu.h",
    "ludcmp": "linear-algebra/solvers/ludcmp/ludcmp.h",
    "mvt": "linear-algebra/kernels/mvt/mvt.h",
    "nussinov": "medley/nussinov/nussinov.h",
    "seidel-2d": "stencils/seidel-2d/seidel-2d.h",
    "symm": "linear-algebra/blas/symm/symm.h",
    "syr2k": "linear-algebra/blas/syr2k/syr2k.h",
    "syrk": "linear-algebra/blas/syrk/syrk.h",
    "trisolv": "linear-algebra/solvers/trisolv/trisolv.h",
    "trmm": "linear-algebra/blas/trmm/trmm.h"
}

# GitHub repository URL
GITHUB_REPO = "https://github.com/MatthiasJReisinger/PolyBenchC-4.2.1.git"
GITHUB_BASE_URL = "https://github.com/MatthiasJReisinger/PolyBenchC-4.2.1/blob/master"

def clone_repository(temp_dir):
    """Clone the PolyBench repository to a temporary directory."""
    repo_path = os.path.join(temp_dir, "PolyBenchC-4.2.1")
    
    print(f"Cloning PolyBench repository to {repo_path}...")
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", GITHUB_REPO, repo_path],
            check=True,
            capture_output=True,
            text=True
        )
        print("✓ Repository cloned successfully")
        return repo_path
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to clone repository: {e}")
        print(f"Error output: {e.stderr}")
        return None

def extract_medium_sizes(header_path):
    """Extract medium dataset sizes from header file."""
    if not os.path.exists(header_path):
        return {}
    
    try:
        with open(header_path, 'r') as f:
            content = f.read()
        
        # Find the MEDIUM_DATASET section
        pattern = r'#\s*ifdef\s+MEDIUM_DATASET\s*(.*?)#\s*endif'
        match = re.search(pattern, content, re.DOTALL)
        
        if not match:
            return {}
        
        medium_section = match.group(1)
        sizes = {}
        
        # Extract #define statements
        define_pattern = r'#\s*define\s+(\w+)\s+(\d+)'
        for define_match in re.finditer(define_pattern, medium_section):
            var_name = define_match.group(1)
            value = define_match.group(2)
            sizes[var_name] = value
            
        return sizes
    
    except Exception as e:
        print(f"Error reading {header_path}: {e}")
        return {}

def get_github_url(file_name):
    """Get GitHub URL for the header file."""
    if file_name in HEADER_FILES:
        return f"{GITHUB_BASE_URL}/{HEADER_FILES[file_name]}"
    return ""

def update_c_file(file_path, medium_sizes, github_url):
    """Update a .c file with medium sizes and GitHub URL comment."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Add GitHub URL comment at the beginning if not already present
        if not content.startswith("// Configuration from:"):
            content = f"// Configuration from: {github_url}\n{content}"
        
        # Update size definitions
        for var_name, value in medium_sizes.items():
            pattern = rf'#define\s+{var_name}\s+\d+'
            replacement = f'#define {var_name} {value}'
            content = re.sub(pattern, replacement, content)
        
        with open(file_path, 'w') as f:
            f.write(content)
        
        return True
    
    except Exception as e:
        print(f"Error updating {file_path}: {e}")
        return False

def main():
    """Main function to process all .c files."""
    # Create temporary directory
    temp_dir = tempfile.mkdtemp(prefix="polybench_update_")
    print(f"Using temporary directory: {temp_dir}")
    
    try:
        # Clone the repository
        repo_path = clone_repository(temp_dir)
        if not repo_path:
            return
        
        # Find the constant directory
        constant_dir = "/home/schrodingerzy/Documents/autolala/analyzer/misc/polybench/polygeist/constant"
        if not os.path.exists(constant_dir):
            print(f"✗ Constant directory not found: {constant_dir}")
            return
        
        # Find all .c files
        c_files = glob.glob(os.path.join(constant_dir, "*.c"))
        
        if not c_files:
            print("✗ No .c files found in the constant directory")
            return
        
        print(f"Found {len(c_files)} .c files to process")
        
        updated_count = 0
        skipped_count = 0
        
        for c_file in c_files:
            file_name = os.path.splitext(os.path.basename(c_file))[0]
            print(f"\nProcessing {file_name}...")
            
            if file_name in HEADER_FILES:
                header_path = os.path.join(repo_path, HEADER_FILES[file_name])
                medium_sizes = extract_medium_sizes(header_path)
                github_url = get_github_url(file_name)
                
                if medium_sizes:
                    if update_c_file(c_file, medium_sizes, github_url):
                        print(f"✓ Updated {file_name} with medium sizes: {', '.join(medium_sizes.keys())}")
                        print(f"  Sizes: {medium_sizes}")
                        updated_count += 1
                    else:
                        print(f"✗ Failed to update {file_name}")
                else:
                    print(f"✗ No medium sizes found for {file_name}")
                    skipped_count += 1
            else:
                print(f"✗ No header file mapping found for {file_name} (skipping)")
                skipped_count += 1
        
        print(f"\nSummary:")
        print(f"✓ Successfully updated: {updated_count} files")
        print(f"⚠ Skipped: {skipped_count} files")
        print(f"📁 Total files processed: {len(c_files)}")
    
    finally:
        # Clean up temporary directory
        print(f"\nCleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir)
        print("✓ Cleanup complete")

if __name__ == "__main__":
    main()
