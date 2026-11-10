#!/bin/bash
cd /workspace/fixes/kaggle-tpu/study
for s in r3s3 r6s2 r3s2 tubes; do
  python3 - "$s" > /tmp/cfg_$s.json <<'PY'
import json,sys,make_cfg as M
s=sys.argv[1]
if s=='tubes': c=dict(mode='tubes',jobs=M.tubes_jobs(),meta={})
else:
    pool=int(s[1]);task='sort'+s[3]
    c=M.cfg('reach',pool,task,0,100)
c.update(out=f'/workspace/fixes/kaggle-tpu/study/out/local-{s}',src='/workspace/fixes/kaggle-tpu/study/ds',procs=7,deadline_s=3*3600)
print(json.dumps(c))
PY
  echo "$(TZ=Europe/Moscow date '+%H:%M:%S МСК') local $s start" >> launcher.log
  STUDY_CFG="$(cat /tmp/cfg_$s.json)" OMP_NUM_THREADS=1 python3 study_core.py > out/local-$s.stdout 2>&1
  echo "$(TZ=Europe/Moscow date '+%H:%M:%S МСК') local $s end: $(tail -1 out/local-$s.stdout)" >> launcher.log
done
