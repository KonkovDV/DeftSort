#!/bin/bash
# Push all shard kernels (retry on session limits), poll, download outputs. Never prints credentials.
K=~/.local/bin/kaggle; D=/workspace/fixes/kaggle-tpu/study; L=$D/launcher.log
cd $D; mkdir -p out state
ts(){ TZ=Europe/Moscow date '+%H:%M:%S МСК'; }
END=$(( $(date +%s) + 4*3600 ))
while [ $(date +%s) -lt $END ]; do
  pending=0
  for d in kernels/*/; do
    sid=$(basename $d); slug=team048-cpu-s-$sid; st=state/$sid
    s=$(cat $st 2>/dev/null)
    case "$s" in
      done|failed) continue;;
      "")
        r=$($K kernels push -p $d 2>&1 | tail -2 | tr '\n' ' ')
        if echo "$r" | grep -qi "successfully\|version"; then echo pushed > $st; echo "$(ts) push $sid OK" >> $L
        else echo "$(ts) push $sid FAIL: ${r:0:200}" >> $L; fi
        pending=1; sleep 3;;
      pushed|running)
        r=$($K kernels status dmitrijkv/$slug 2>&1 | tail -1)
        if echo "$r" | grep -q COMPLETE; then
          mkdir -p out/$sid; $K kernels output dmitrijkv/$slug -p out/$sid -o >/dev/null 2>&1
          echo done > $st; echo "$(ts) $sid COMPLETE, $(wc -l < out/$sid/results.jsonl 2>/dev/null) rows; $(cat out/$sid/status.json 2>/dev/null | head -c 150)" >> $L
        elif echo "$r" | grep -q "ERROR\|CANCEL"; then
          mkdir -p out/$sid; $K kernels output dmitrijkv/$slug -p out/$sid -o >/dev/null 2>&1
          echo failed > $st; echo "$(ts) $sid ERROR: $r" >> $L
        else
          echo running > $st; pending=1
        fi;;
    esac
  done
  [ $pending -eq 0 ] && { echo "$(ts) ALL SHARDS FINISHED" >> $L; break; }
  sleep 60
done
