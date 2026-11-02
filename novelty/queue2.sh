#!/bin/sh
cd /home/claude/repro
R=results/novelty
python3 novelty/partB.py full >> $R/partB_full.log 2>&1; echo "B full exit $? $(date)" >> $R/queue.log
python3 novelty/partB.py randsplit > $R/partB_rand.log 2>&1; echo "B rand exit $? $(date)" >> $R/queue.log
python3 novelty/partD.py arch > $R/partD_arch.log 2>&1; echo "D arch exit $? $(date)" >> $R/queue.log
python3 novelty/partA_recal.py > $R/partA_recal.log 2>&1; echo "A recal exit $? $(date)" >> $R/queue.log
python3 novelty/partB_sens.py > $R/partB_sens.log 2>&1; echo "B sens exit $? $(date)" >> $R/queue.log
echo ALL DONE >> $R/queue.log
