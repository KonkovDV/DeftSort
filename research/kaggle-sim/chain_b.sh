#!/bin/bash
K=~/.local/bin/kaggle
while [ ! -f out_v9a/.done ]; do sleep 60; done
q=$($K quota 2>&1 | awk '/^GPU/{gsub("h","",$3);print $3}')
echo "A done $(date +%H:%M) quota=$q"
if awk "BEGIN{exit !($q>=1.35)}"; then p=v9/kB_s75; elif awk "BEGIN{exit !($q>=1.0)}"; then p=v9/kB_s55; else echo "quota too low"; exit 0; fi
$K kernels push -p $p 2>&1 | grep -vi token | tail -1; echo "pushed $p"
mkdir -p out_v9b2
while true; do st=$($K kernels status dmitrijkv/team048-sim-v9b 2>&1|tail -1); echo "$(date +%H:%M) $st" >> watch_v9b2.log
 echo "$st" | grep -qE 'COMPLETE|ERROR|CANCEL' && break; sleep 120; done
$K kernels output dmitrijkv/team048-sim-v9b -p out_v9b2 >/dev/null 2>&1; echo "B finished: $st"; cat out_v9b2/results.jsonl 2>/dev/null
