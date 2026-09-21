#!/bin/bash
cd /Users/shivenkhurana/bioproj
st() { kaggle kernels status blackpearls1/tda-egnn-$1 2>&1 | tail -1; }
# wait for a free GPU slot
for i in $(seq 1 200); do
  a=$(st phase7-baseline-fpocket); b=$(st phase8-sweep-gates)
  ra=$(echo "$a" | grep -c RUNNING); rb=$(echo "$b" | grep -c RUNNING)
  if [ $((ra+rb)) -lt 2 ]; then echo "slot free after ~$((i*2))min"; break; fi
  sleep 120
done
echo "P7: $a"; echo "P8: $b"
kaggle kernels push -p push/phase9 2>&1 | tail -2
# now wait for everything
for i in $(seq 1 240); do
  a=$(st phase7-baseline-fpocket); b=$(st phase8-sweep-gates); c=$(st phase9-mcsa-catalytic)
  n=$(echo "$a$b$c" | grep -o RUNNING | wc -l)
  q=$(echo "$a$b$c" | grep -o QUEUED | wc -l)
  if [ $((n+q)) -eq 0 ]; then echo "ALL DONE after ~$((i*2))min"; echo "$a"; echo "$b"; echo "$c"; exit 0; fi
  sleep 120
done
echo TIMEOUT
