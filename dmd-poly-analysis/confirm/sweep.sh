#!/bin/bash
cg(){ valgrind --tool=cachegrind --cache-sim=yes --D1=32768,8,64 --LL=2097152,16,64 \
   --cachegrind-out-file=/dev/null ./k "$1" "$2" 2>&1 \
   | grep -E "D refs:|D1  misses:|LLd misses:" | grep -oE "[0-9,]+" | tr -d ',' | paste -sd' '; }
echo "kernel N Drefs D1miss LLdmiss D1rate% LLrate%"
for kv in mm_naive:128,256,384,512 mm_tiled:128,256,384,512 syrk_naive:128,256,384,512 mvt_naive:512,1024,2048 mvt_interch:512,1024,2048; do
  k=${kv%%:*}; ns=${kv#*:}
  for N in ${ns//,/ }; do
    read DR D1 LL <<< "$(cg $N $k)"
    d1r=$(python3 -c "print(f'{100*$D1/$DR:.2f}')"); llr=$(python3 -c "print(f'{100*$LL/$DR:.3f}')")
    echo "$k $N $DR $D1 $LL $d1r $llr"
  done
done
echo CGDONE
