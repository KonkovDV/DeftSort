#!/bin/bash
# box watcher: wait for v3 to end -> download out_v3 -> push v4 (retry on GPU-slot limit) -> poll v4 -> download out_v4
export PATH=$HOME/.local/bin:$PATH; K=dmitrijkv/team048-sim-stage34; D=/workspace/fixes/kaggle-sim; L=$D/watch.log
st(){ kaggle kernels status $K 2>&1|grep -o 'Status\.[A-Z_]*'; }
log(){ echo "$(date '+%F %H:%M MSK') $*" >> $L; }
log "watch start"
while [ "$(st)" = Status.RUNNING ] || [ "$(st)" = Status.QUEUED ]; do sleep 120; done
log "v3 ended: $(st)"; mkdir -p $D/out_v3; timeout 900 kaggle kernels output $K -p $D/out_v3 -o >> $L 2>&1; log "out_v3 files: $(ls $D/out_v3|wc -l)"
until timeout 180 kaggle kernels push -p $D/v4/k >> $L 2>&1 && ! tail -1 $L|grep -qi error; do log "push retry"; sleep 300; done
log "v4 pushed"; sleep 60
while [ "$(st)" = Status.RUNNING ] || [ "$(st)" = Status.QUEUED ]; do sleep 120; done
log "v4 ended: $(st)"; mkdir -p $D/out_v4; timeout 900 kaggle kernels output $K -p $D/out_v4 -o >> $L 2>&1; log "out_v4 files: $(ls $D/out_v4|wc -l)"
