#!/bin/bash -l
#$ -cwd
#$ -o log/qsub_log_o.$JOB_ID.$TASK_ID
#$ -e log/qsub_log_e.$JOB_ID.$TASK_ID
pyfile=$1
plistfile=$2
t=$SGE_TASK_ID
params_command="sed -n '"$t"p' $plistfile"
params=`eval $params_command`
array=(${params//;/ })
param_string=`( IFS=$' '; echo "${array[*]}" )`
########## cell segmentation #########
echo $param_string 
reuse Anaconda3
source activate /stanley/WangLab/envs/atlas 
python $pyfile $param_string
echo "Finished"
now=$(date +"%T")
echo "Current time : $now"
