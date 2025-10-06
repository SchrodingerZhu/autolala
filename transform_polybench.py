#!/usr/bin/env python3
"""
Script to transform PolyBench C files by moving array parameters to global volatile declarations.
"""

import os
import re
import glob
import argparse
from typing import List, Tuple, Dict

def parse_function_signature(line: str) -> Tuple[str, List[str], List[str]]:
    """
    Parse a function signature and extract function name, array parameters, and non-array parameters.
    
    Returns:
        - function_name: The name of the function
        - array_params: List of array parameter declarations
        - non_array_params: List of non-array parameter declarations
    """
    # Match function signature pattern
    func_match = re.match(r'void\s+(\w+)\s*\((.*)\)\s*{?', line)
    if not func_match:
        return "", [], []
    
    func_name = func_match.group(1)
    params_str = func_match.group(2).strip()
    
    if not params_str:
        return func_name, [], []
    
    # Split parameters by comma, but be careful with array dimensions
    params = []
    current_param = ""
    bracket_depth = 0
    
    for char in params_str:
        if char == '[':
            bracket_depth += 1
        elif char == ']':
            bracket_depth -= 1
        elif char == ',' and bracket_depth == 0:
            params.append(current_param.strip())
            current_param = ""
            continue
        current_param += char
    
    if current_param.strip():
        params.append(current_param.strip())
    
    array_params = []
    non_array_params = []
    
    for param in params:
        param = param.strip()
        if '[' in param and ']' in param:
            array_params.append(param)
        else:
            non_array_params.append(param)
    
    return func_name, array_params, non_array_params

def extract_array_name(param: str) -> str:
    """Extract variable name from array parameter declaration."""
    # Match pattern like "DATA_TYPE A[M][N]" to extract "A"
    match = re.search(r'(\w+)\s*\[', param)
    return match.group(1) if match else ""

def transform_file(input_path: str, output_path: str = None) -> bool:
    """
    Transform a single C file by moving array parameters to global declarations.
    
    Args:
        input_path: Path to input C file
        output_path: Path to output C file (if None, overwrites input)
    
    Returns:
        True if file was modified, False otherwise
    """
    if output_path is None:
        output_path = input_path
    
    try:
        with open(input_path, 'r') as f:
            lines = f.readlines()
    except IOError as e:
        print(f"Error reading {input_path}: {e}")
        return False
    
    modified = False
    output_lines = []
    global_declarations = []
    
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        
        # Check if this line contains a kernel function signature
        if re.match(r'void\s+kernel_\w+\s*\(', line):
            # Handle multi-line function signatures
            full_signature = line
            j = i + 1
            
            # Continue reading until we find the opening brace
            while j < len(lines) and '{' not in full_signature:
                full_signature += ' ' + lines[j].strip()
                j += 1
            
            # Parse the complete function signature
            func_name, array_params, non_array_params = parse_function_signature(full_signature)
            
            if array_params:
                modified = True
                
                # Generate global volatile declarations
                for param in array_params:
                    global_decl = f"volatile {param};"
                    global_declarations.append(global_decl)
                
                # Generate new function signature
                if non_array_params:
                    new_signature = f"void {func_name}({', '.join(non_array_params)}) {{"
                else:
                    new_signature = f"void {func_name}() {{"
                
                output_lines.append(new_signature)
                
                # Skip the original lines that made up the function signature
                i = j
                continue
        
        output_lines.append(line)
        i += 1
    
    if modified:
        # Insert global declarations after the #define statements
        final_output = []
        defines_ended = False
        
        for line in output_lines:
            final_output.append(line)
            
            # Insert global declarations after the last #define and before the first function
            if not defines_ended and line.strip() and not line.startswith('#'):
                if line.startswith('void '):
                    # Insert globals before this function
                    final_output.pop()  # Remove the function line temporarily
                    final_output.append("")  # Add blank line
                    for global_decl in global_declarations:
                        final_output.append(global_decl)
                    final_output.append("")  # Add blank line
                    final_output.append(line)  # Add the function line back
                    defines_ended = True
        
        # Write the transformed file
        try:
            with open(output_path, 'w') as f:
                for line in final_output:
                    f.write(line + '\n')
            print(f"Transformed: {input_path}")
            return True
        except IOError as e:
            print(f"Error writing {output_path}: {e}")
            return False
    
    return False

def main():
    parser = argparse.ArgumentParser(
        description="Transform PolyBench C files by moving array parameters to global volatile declarations"
    )
    parser.add_argument(
        "directory",
        help="Directory containing C files to transform"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be transformed without making changes"
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for transformed files (default: overwrite input files)"
    )
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.directory):
        print(f"Error: {args.directory} is not a valid directory")
        return 1
    
    # Find all C files in the directory
    c_files = glob.glob(os.path.join(args.directory, "*.c"))
    
    if not c_files:
        print(f"No C files found in {args.directory}")
        return 1
    
    print(f"Found {len(c_files)} C files")
    
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    transformed_count = 0
    
    for c_file in sorted(c_files):
        output_file = None
        if args.output_dir:
            filename = os.path.basename(c_file)
            output_file = os.path.join(args.output_dir, filename)
        
        if args.dry_run:
            # Just analyze without writing
            with open(c_file, 'r') as f:
                content = f.read()
            
            if re.search(r'void\s+kernel_\w+\s*\([^)]*DATA_TYPE.*\[.*\]', content):
                print(f"Would transform: {c_file}")
                transformed_count += 1
        else:
            if transform_file(c_file, output_file):
                transformed_count += 1
    
    if args.dry_run:
        print(f"\nWould transform {transformed_count} files")
    else:
        print(f"\nTransformed {transformed_count} files")
    
    return 0

if __name__ == "__main__":
    exit(main())
