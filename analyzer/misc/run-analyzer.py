#!/usr/bin/env python3

import os
import subprocess
import json
import glob
from fractions import Fraction

def parse_fraction(fraction_str):
    """Parse a fraction string like '\\frac{3447}{4}' or simple numbers"""
    fraction_str = fraction_str.strip()
    
    # Handle simple numbers
    try:
        return float(fraction_str)
    except ValueError:
        pass
    
    # Handle LaTeX fractions like \frac{3447}{4}
    if fraction_str.startswith('\\frac{'):
        # Extract numerator and denominator
        content = fraction_str[6:-1]  # Remove \frac{ and }
        parts = content.split('}{')
        if len(parts) == 2:
            numerator = parts[0]
            denominator = parts[1]
            
            # Handle cases like \frac{1}{4} \left(8 i3+1679\right)
            if '\\left(' in denominator:
                denominator = denominator.split('\\left(')[0].strip()
            
            try:
                return float(Fraction(int(numerator), int(denominator)))
            except:
                pass
    
    # Handle expressions with variables (return 0 as fallback)
    return 0.0

def find_closest_turning_point_index(turning_points, target=4096):
    """Find the index of the turning point value closest to and <= target"""
    closest_index = -1
    closest_value = -1
    
    for i, tp in enumerate(turning_points):
        if tp <= target and tp > closest_value:
            closest_value = tp
            closest_index = i
    
    return closest_index 

def extract_json_from_output(output):
    """Extract JSON from output starting from the first '{'"""
    json_start = output.find('{')
    if json_start == -1:
        return None
    
    json_part = output[json_start:]
    try:
        return json.loads(json_part)
    except json.JSONDecodeError:
        # Try to find the end of JSON by counting braces
        brace_count = 0
        for i, char in enumerate(json_part):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    try:
                        return json.loads(json_part[:i+1])
                    except json.JSONDecodeError:
                        continue
        return None

def run_analyzer(file_path, approximation_method):
    """Run the analyzer command and return parsed JSON output"""
    cmd = f'cargo run --release --bin analyzer -- -i {file_path} -m /tmp/test.svg --json barvinok --block-size=8 --barvinok-arg="--approximation-method={approximation_method}"'

    if approximation_method == "":
        cmd = f'cargo run --release --bin analyzer -- -i {file_path} -m /tmp/test.svg --json barvinok --block-size=8'
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            return extract_json_from_output(result.stdout)
        else:
            print(f"Error running command for {file_path}: {result.stderr}")
            return None
    except subprocess.TimeoutExpired:
        print(f"Timeout running command for {file_path}")
        return None
    except Exception as e:
        print(f"Unexpected error for {file_path}: {e}")
        return None

def process_file(file_path, approximation_methods):
    """Process a single file with all approximation methods"""
    filename = os.path.basename(file_path)
    results = []
    
    for method in approximation_methods:
        print(f"Processing {filename} with method {method}...")
        
        data = run_analyzer(file_path, method)
        if not data:
            continue
            
        # Extract data from JSON
        miss_ratio_curve = data.get('miss_ratio_curve', {})
        turning_points = miss_ratio_curve.get('turning_points', [])
        miss_ratios = miss_ratio_curve.get('miss_ratio', [])

        total_count = data.get('total_count', '0')
        
        if not turning_points or not miss_ratios:
            print(f"Missing data in output for {filename} with method {method}")
            continue
        
        # Find closest turning point index <= 4096
        closest_index = find_closest_turning_point_index(turning_points, 4096 // 8 // 8)
        
        if closest_index == -1 or closest_index >= len(miss_ratios):
            print(f"No valid turning point found for {filename} with method {method}")
            continue
        
        # Get corresponding miss ratio (it's already a decimal)
        miss_ratio = miss_ratios[closest_index] if closest_index else miss_ratios[0]
        try:
            total_count_num = float(total_count)
            # Calculate total miss count: miss_ratio * total_count
            final_value = miss_ratio * total_count_num
            results.append((method, final_value))
            
        except ValueError:
            print(f"Error parsing total_count for {filename} with method {method}")
            continue
    
    return filename, results

def main():
    # Configuration
    input_dir = "./analyzer/misc/polybench/const/"
    approximation_methods = [""]
    
    # Find all files in the directory
    if not os.path.exists(input_dir):
        print(f"Directory {input_dir} does not exist!")
        return
    
    files = []
    for ext in ['*', '*.c', '*.cpp', '*.h']:  # Add more extensions as needed
        files.extend(glob.glob(os.path.join(input_dir, ext)))
    
    # Filter out directories
    files = [f for f in files if os.path.isfile(f)]
    #files = files[:1]
    if not files:
        print(f"No files found in {input_dir}")
        return
    
    print(f"Found {len(files)} files to process")
    
    # Process each file
    all_results = []
    
    for file_path in files:
        if not file_path.endswith('.mlir') or 'fdtd-apml' in file_path:
            continue
        filename, results = process_file(file_path, approximation_methods)
        
        if results:
            for method, value in results:
                all_results.append(f"{filename}_{method}, {value}")
        else:
            print(f"No results for {filename}")
    
    # Save results to file
    with open('result.txt', 'w') as f:
        f.write("filename, value\n")
        for result in all_results:
            f.write(result + '\n')
    
    print(f"Results saved to result.txt")
    print(f"Total results: {len(all_results)}")

if __name__ == "__main__":
    main()
