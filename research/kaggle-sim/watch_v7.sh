#!/bin/bash
# waits for kernel v7 (pushed 2026-11-09 09:59 MSK) to finish, downloads output to out_v7 (never prints the token)
export PATH=$HOME/.local/bin:$PATH; K=dmitrijkv/team048-sim-stage34; D=/workspace/fixes/kaggle-sim; L=$D/watch.log
st(){ timeout 60 kaggle kernels status $K 2>&1|grep -o 'Status\.[A-Z_]*\|COMPLETE\|ERROR\|RUNNING\|QUEUED'|head -1; }
sleep 120; S=$(st); n=0
while [ "$S" != "Status.COMPLETE" ] && [ "$S" != "Status.ERROR" ] && [ "$S" != "COMPLETE" ] && [ "$S" != "ERROR" ] && [ $n -lt 80 ]; do sleep 120; S=$(st); n=$((n+1)); done
echo "$(date '+%F %H:%M MSK') v7 ended: $S" >> $L; mkdir -p $D/out_v7; timeout 900 kaggle kernels output $K -p $D/out_v7 -o >> $L 2>&1
echo "$(date '+%F %H:%M MSK') v7 output downloaded: $(ls $D/out_v7 | wc -l) entries" >> $L
