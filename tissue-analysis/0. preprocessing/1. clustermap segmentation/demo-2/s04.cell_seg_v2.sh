#!/bin/bash -l
# To run this script, submit with UGER with one argument (run_id, corresponding to name of plist file, starting with test if test). 
# All other parameters are provided by the plist

# Submit segmentation tasks
run_id=$1

#$ -cwd
#$ -o log/qsub_log_o.$JOB_ID.$TASK_ID
#$ -e log/qsub_log_e.$JOB_ID.$TASK_ID

plistfile="01.plist/$run_id"
t=$SGE_TASK_ID 
params_command="sed -n '"$t"p' $plistfile"
params=`eval $params_command`
array=(${params//;/ })
param_string=`( IFS=$' '; echo "${array[*]}" )`

########## cell segmentation #########
echo $param_string 
reuse Anaconda3
source activate /stanley/WangLab/envs/atlas
python s04.cell_seg_v2.py $param_string $run_id
echo "Finished"
now=$(date +"%T")
echo "Current time : $now"
