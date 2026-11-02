#!/bin/sh
cd /home/claude/repro
R=results/novelty
until grep -q "PART A TRAINING DONE" $R/partA/stdout.log 2>/dev/null; do sleep 30; done
echo "A done $(date)" >> $R/queue.log
python3 novelty/partB.py small > $R/partB_small.log 2>&1; echo "B small exit $? $(date)" >> $R/queue.log
python3 novelty/partB.py chem > $R/partB_chem.log 2>&1; echo "B chem exit $? $(date)" >> $R/queue.log
python3 novelty/partC.py > $R/partC.log 2>&1; echo "C exit $? $(date)" >> $R/queue.log
python3 novelty/partC_sat.py > $R/partC_sat.log 2>&1; echo "C sat exit $? $(date)" >> $R/queue.log
python3 novelty/partD.py chem > $R/partD_chem.log 2>&1; echo "D chem exit $? $(date)" >> $R/queue.log
python3 novelty/partB.py full > $R/partB_full.log 2>&1; echo "B full exit $? $(date)" >> $R/queue.log
python3 novelty/partB.py randsplit > $R/partB_rand.log 2>&1; echo "B rand exit $? $(date)" >> $R/queue.log
python3 novelty/partD.py arch > $R/partD_arch.log 2>&1; echo "D arch exit $? $(date)" >> $R/queue.log
python3 novelty/partA_recal.py > $R/partA_recal.log 2>&1; echo "A recal exit $? $(date)" >> $R/queue.log
python3 novelty/partB_sens.py > $R/partB_sens.log 2>&1; echo "B sens exit $? $(date)" >> $R/queue.log
echo ALL DONE >> $R/queue.log
