#!/bin/bash
K=~/.local/bin/kaggle
for s in a b; do done_[$s]=0; done
while true; do
  all=1
  for s in a; do
    d=out_v9$s; [ -f $d/.done ] && continue
    st=$($K kernels status dmitrijkv/team048-sim-v9$s 2>&1 | tail -1)
    echo "$(date '+%H:%M:%S') v9$s $st" >> watch_v9.log
    if echo "$st" | grep -qE 'COMPLETE|ERROR|CANCEL'; then mkdir -p $d && $K kernels output dmitrijkv/team048-sim-v9$s -p $d >> watch_v9.log 2>&1; touch $d/.done
    else all=0; fi
  done
  [ $all = 1 ] && break
  sleep 120
done
echo ALLDONE >> watch_v9.log
