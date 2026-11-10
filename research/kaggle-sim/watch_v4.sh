#!/bin/bash
export PATH=$HOME/.local/bin:$PATH; K=dmitrijkv/team048-sim-stage34; D=/workspace/fixes/kaggle-sim; L=$D/watch.log
st(){ timeout 60 kaggle kernels status $K 2>&1|grep -o 'Status\.[A-Z_]*'; }
while [ "$(st)" = Status.RUNNING ] || [ "$(st)" = Status.QUEUED ]; do sleep 120; done
echo "$(date '+%F %H:%M MSK') v5 ended: $(st)" >> $L; mkdir -p $D/out_v4; timeout 900 kaggle kernels output $K -p $D/out_v4 -o >> $L 2>&1
