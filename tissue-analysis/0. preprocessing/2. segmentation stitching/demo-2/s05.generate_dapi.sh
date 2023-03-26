#!/bin/bash

# Request mem
#$ -l h_vmem=5G

# Request runtime
#$ -l h_rt=00:10:00

# Specify output file destination -- CHANGE ONCE COPIED TO TISSUE DIR
#$ -cwd
#$ -o log/dapi/qsub_log_o.$JOB_ID.$TASK_ID
#$ -e log/dapi/qsub_log_e.$JOB_ID.$TASK_ID


# Set up python environment
source /broad/software/scripts/useuse
reuse Anaconda3
source activate /stanley/WangLab/envs/atlas

# Set up arguments
position_num=$SGE_TASK_ID
sample=$1
tissue=`echo $1 | cut -f1 -d '_'`

# Run script
python /stanley/WangLab/atlas/code/3.stitch/s05.generate_maxrotatedapi.py -s $sample -p $position_num 

