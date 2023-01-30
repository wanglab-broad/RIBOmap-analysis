#!/bin/bash -l 

# Submit global registration tasks
plist=$1
mkdir -p -m775 log/global

#$ -cwd
#$ -o log/global/qsub_log_o.$JOB_ID.$TASK_ID
#$ -e log/global/qsub_log_e.$JOB_ID.$TASK_ID

core_path="/stanley/WangLab/atlas/code/1.registration"
t=$SGE_TASK_ID
params_command="sed -n '"$t"p' $plist"
params=`eval $params_command`
array=(${params//;/ })
tile=${array[0]}
sqrt_pieces=${array[1]}
mode=${array[2]}
t_para=${array[3]}
input_dim=${array[4]} 
run_id=${array[5]} 

source "/broad/software/scripts/useuse"
reuse Matlab
now=$(date +"%T")
echo "Current time : $now"
command="addpath('$core_path');core_matlab('$tile',$sqrt_pieces,'$mode',$t_para,$input_dim,'$run_id');exit;"
echo $command
matlab -nodisplay -nosplash -nodesktop -r $command 
echo "Finished"
now=$(date +"%T")
echo "Current time : $now"

