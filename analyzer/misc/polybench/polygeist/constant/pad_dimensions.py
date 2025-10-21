#!/usr/bin/env python3
"""
Script to pad array last dimensions to be multiples of both 12 and 8 (i.e., multiples of 24).
For each array declaration, finds the last dimension and pads it up to the next multiple of 24.
"""

import re
from pathlib import Path
from math import gcd, lcm

def find_next_multiple(value, multiple):
    """Find the next number >= value that's a multiple of 'multiple'."""
    if value % multiple == 0:
        return value
    return ((value // multiple) + 1) * multiple

def parse_array_declarations(content):
    """Parse all volatile array declarations and extract dimensions."""
    # Pattern: volatile DATA_TYPE name[dim1][dim2]...[dimN];
    pattern = r'volatile\s+DATA_TYPE\s+(\w+)((?:\[\w+\])+);'
    matches = re.finditer(pattern, content)
    
    arrays = []
    for match in matches:
        name = match.group(1)
        dims_str = match.group(2)
        dims = re.findall(r'\[(\w+)\]', dims_str)
        arrays.append({
            'name': name,
            'dims': dims,
            'full_match': match.group(0)
        })
    
    return arrays

def parse_defines(content):
    """Parse all #define SIZE_NAME value pairs."""
    pattern = r'#define\s+(\w+)\s+(\d+)'
    matches = re.finditer(pattern, content)
    
    defines = {}
    for match in matches:
        name = match.group(1)
        value = int(match.group(2))
        defines[name] = value
    
    return defines

def transform_file(input_path):
    """Transform a single C file to pad last dimensions."""
    with open(input_path, 'r') as f:
        content = f.read()
    
    # Parse defines
    defines = parse_defines(content)
    
    # Parse array declarations
    arrays = parse_array_declarations(content)
    
    if not arrays:
        print(f"  No arrays found in {input_path.name}")
        return
    
    # LCM of 12 and 8
    target_multiple = lcm(12, 8)  # = 24
    
    # Collect replacements
    replacements = []
    
    for array in arrays:
        if not array['dims']:
            continue
        
        last_dim_name = array['dims'][-1]
        
        # Skip if it's a number literal (shouldn't happen in our case)
        if last_dim_name.isdigit():
            continue
        
        # Get the value of this dimension
        if last_dim_name not in defines:
            print(f"  Warning: {last_dim_name} not found in defines for array {array['name']}")
            continue
        
        original_value = defines[last_dim_name]
        padded_value = find_next_multiple(original_value, target_multiple)
        
        if original_value == padded_value:
            continue  # Already a multiple
        
        # Build new array declaration with padded last dimension
        dims_except_last = array['dims'][:-1]
        new_dims_str = ''.join(f'[{d}]' for d in dims_except_last) + f'[{padded_value}]'
        new_decl = f"volatile DATA_TYPE {array['name']}{new_dims_str};"
        
        replacements.append({
            'old': array['full_match'],
            'new': new_decl,
            'array': array['name'],
            'dim': last_dim_name,
            'original': original_value,
            'padded': padded_value
        })
    
    if not replacements:
        print(f"  {input_path.name}: All dimensions already multiples of {target_multiple}")
        return
    
    # Apply replacements
    for repl in replacements:
        content = content.replace(repl['old'], repl['new'])
        print(f"  {input_path.name}: {repl['array']} last dim {repl['dim']} {repl['original']} → {repl['padded']}")
    
    # Write back
    with open(input_path, 'w') as f:
        f.write(content)

def main():
    script_dir = Path(__file__).parent
    
    # Process all .c files in the constant_global directory
    c_files = list(script_dir.glob('*.c'))
    
    if not c_files:
        print("No .c files found in the current directory")
        return
    
    print(f"Padding last dimensions to multiples of {lcm(12, 8)} (LCM of 12 and 8)")
    print(f"Found {len(c_files)} C files to process")
    print("-" * 60)
    
    for c_file in sorted(c_files):
        transform_file(c_file)
    
    print("-" * 60)
    print("Padding complete!")

if __name__ == '__main__':
    main()
