#!/bin/bash

# Request mem
#$ -l h_vmem=15G

# Request runtime
#$ -l h_rt=5:00:00

# Specify output file destination
#$ -cwd
mkdir -p log/stitch 
#$ -o log/stitch/qsub_log_o.$JOB_ID
#$ -e log/stitch/qsub_log_e.$JOB_ID

# Set up python environment
source /broad/software/scripts/useuse
reuse Anaconda3
source activate /stanley/WangLab/envs/atlas

# Set up arguments
sample_id=$1
suffix=$2
trial=$3

# Run script
python /stanley/WangLab/atlas/Brain/00.sh/03.stitch/s06.stitch_unassigned_reads.py $sample_id $suffix $trial
