#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import argparse
import tempfile
import os
import json
import time
from tqdm.contrib.concurrent import process_map

def execute(args):
    src, blocks, block_size = args
    with tempfile.TemporaryDirectory() as dir:
        subprocess.run([
            'clang', '-static', '-nostdlib', '-fno-stack-protector', '-fno-pic', '-O3', src, '-o', 'a.out'], check=True, cwd=dir)
        out = subprocess.run(['valgrind', '--tool=cachegrind', '--cache-sim=yes',
                        '--D1=' + str(blocks * block_size) + ',' + str(blocks) + ',' + str(block_size), './a.out'], 
                        check=True, capture_output=True, cwd=dir)
        drefs = 0
        d1_miss = 0
        for line in out.stderr.decode().split('\n'):
            if 'D1  misses' in line:
                d1_miss = int(line.split()[3].replace(',', ''))
            elif 'D refs' in line:
                drefs = int(line.split()[3].replace(',', ''))
        return d1_miss / drefs
        

def main():
    parser = argparse.ArgumentParser(description='miss ratio curve generator')
    parser.add_argument("--src", type=str, help="source code", required=True)
    parser.add_argument('--max-cache', type=int, default=1024, help='maximum number of cache blocks')
    parser.add_argument('--block', type=int, default=32, help='block size')
    parser.add_argument('--step', type=int, default=1, help='step size')
    parser.add_argument('--output', type=str, default='/tmp/miss_ratio.json', help='output file')
    args = parser.parse_args()
    src = os.path.abspath(args.src)
    all_task_args = []
    for blocks in range(2, args.max_cache + 1, args.step):
        all_task_args.append((src, blocks, args.block))
    start_time = time.time()
    results = process_map(execute, all_task_args, chunksize=1, max_workers=os.cpu_count(), desc='Running tasks', unit='task')
    end_time = time.time()
    map = {}
    map['miss_ratio'] = results
    map['blocks'] = list(range(2, args.max_cache + 1, args.step))
    map['time_elapsed'] = end_time - start_time
    with open(args.output, 'w') as f:
        json.dump(map, f, indent=4)

if __name__ == '__main__':
    main()
