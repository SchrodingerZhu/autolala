#!/usr/bin/env python3
"""
Script to port einsum/constant programs to use global memory like polybench format.
Changes:
1. Convert function parameters to global volatile arrays
2. Change DATA_TYPE from float to double
3. Update float literals (.0f -> .0)
4. Make kernel function take no parameters
"""

import re
import sys
from pathlib import Path

def parse_function_signature(content):
    """Extract function signature and parameters."""
    # Match function definition with parameters
    func_pattern = r'void\s+(\w+)\s*\((.*?)\)\s*\{'
    match = re.search(func_pattern, content, re.DOTALL)
    if not match:
        return None, None, None
    
    func_name = match.group(1)
    params_str = match.group(2)
    func_start = match.start()
    
    # Parse parameters
    params = []
    if params_str.strip():
        # Split by comma, but handle multi-dimensional arrays carefully
        param_parts = []
        bracket_depth = 0
        current = ""
        for char in params_str:
            if char == '[':
                bracket_depth += 1
            elif char == ']':
                bracket_depth -= 1
            elif char == ',' and bracket_depth == 0:
                param_parts.append(current.strip())
                current = ""
                continue
            current += char
        if current.strip():
            param_parts.append(current.strip())
        
        for param in param_parts:
            param = param.strip()
            if not param:
                continue
            
            # Parse: [const] DATA_TYPE name[dims...]
            # Remove const if present
            param = re.sub(r'\bconst\s+', '', param)
            
            # Extract array name and dimensions
            # Pattern: DATA_TYPE name[dim1][dim2]... or DATA_TYPE *name
            if '*' in param:
                # Pointer type (like *trace)
                match = re.match(r'DATA_TYPE\s+\*(\w+)', param)
                if match:
                    params.append({
                        'name': match.group(1),
                        'type': 'DATA_TYPE',
                        'dims': [],
                        'is_pointer': True
                    })
            else:
                # Array type
                match = re.match(r'DATA_TYPE\s+(\w+)((?:\[\w+\])*)', param)
                if match:
                    name = match.group(1)
                    dims_str = match.group(2)
                    dims = re.findall(r'\[(\w+)\]', dims_str)
                    params.append({
                        'name': name,
                        'type': 'DATA_TYPE',
                        'dims': dims,
                        'is_pointer': False
                    })
    
    return func_name, params, func_start

def generate_global_declarations(params):
    """Generate global volatile declarations from parameters."""
    declarations = []
    for param in params:
        if param['is_pointer']:
            # Pointer becomes a scalar volatile variable
            declarations.append(f"volatile {param['type']} {param['name']};")
        else:
            # Array becomes volatile array
            dims_str = ''.join(f"[{dim}]" for dim in param['dims'])
            declarations.append(f"volatile {param['type']} {param['name']}{dims_str};")
    return declarations

def transform_file(input_path, output_path):
    """Transform a single C file."""
    with open(input_path, 'r') as f:
        content = f.read()
    
    # Change DATA_TYPE from float to double
    content = re.sub(r'#define\s+DATA_TYPE\s+float', '#define DATA_TYPE double', content)
    
    # Change float literals to double (.0f -> .0, handle 0.0f as well)
    content = re.sub(r'(\d+\.?\d*)f\b', r'\1', content)
    
    # Parse function signature
    func_name, params, func_start = parse_function_signature(content)
    
    if not func_name or not params:
        print(f"Warning: Could not parse function in {input_path}")
        # Still write the file with type changes
        with open(output_path, 'w') as f:
            f.write(content)
        return
    
    # Generate global declarations
    global_decls = generate_global_declarations(params)
    
    # Find where to insert globals (after #define statements)
    define_pattern = r'((?:#define.*\n)+)\n*'
    define_match = re.search(define_pattern, content)
    
    if define_match:
        insert_pos = define_match.end()
        # Insert global declarations
        globals_text = '\n\n' + '\n'.join(global_decls) + '\n'
        content = content[:insert_pos] + globals_text + content[insert_pos:]
        
        # Update function start position
        func_start += len(globals_text)
    
    # Replace function signature to remove parameters
    # Find the function signature again in the updated content
    func_pattern = r'void\s+' + re.escape(func_name) + r'\s*\([^)]*\)\s*\{'
    content = re.sub(func_pattern, f'void {func_name}() {{', content)
    
    # Fix pointer dereferences - convert *varname to varname for variables that were pointers
    for param in params:
        if param['is_pointer']:
            # Replace *varname with varname (but not in declarations)
            # This handles cases like *trace = value or *trace += value
            content = re.sub(rf'\*{param["name"]}\b', param['name'], content)
    
    # Write output
    with open(output_path, 'w') as f:
        f.write(content)
    
    print(f"Transformed: {input_path.name} -> {output_path.name}")

def main():
    script_dir = Path(__file__).parent
    input_dir = script_dir
    output_dir = script_dir.parent / 'constant_global'
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    
    # Process all .c files in the constant directory
    c_files = list(input_dir.glob('*.c'))
    
    if not c_files:
        print("No .c files found in the current directory")
        return
    
    print(f"Found {len(c_files)} C files to process")
    print(f"Output directory: {output_dir}")
    print("-" * 60)
    
    for c_file in sorted(c_files):
        output_file = output_dir / c_file.name
        transform_file(c_file, output_file)
    
    print("-" * 60)
    print(f"Transformation complete! Files written to: {output_dir}")

if __name__ == '__main__':
    main()
