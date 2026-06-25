#!/bin/bash
cg(){ valgrind --tool=cachegrind --cache-sim=yes --D1=32768,8,64 --LL=2097152,16,64 \
   --cachegrind-out-file=/dev/null ./k "$1" "$2" 2>&1 \
   | grep -E "D refs|D1  misses|LLd misses" \
   | sed 's/[ ,]//g;s/(.*)//'; }
