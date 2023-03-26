#!/bin/bash -l
ppath="/stanley/WangLab/jiahao/Github/RIBOmap/cluster-ribomap-2"
#$ -o /stanley/WangLab/jiahao/Github/RIBOmap/cluster-ribomap-2/log/qsub_log_o.$JOB_ID.$TASK_ID
#$ -e /stanley/WangLab/jiahao/Github/RIBOmap/cluster-ribomap-2/log/qsub_log_e.$JOB_ID.$TASK_ID

source "/broad/software/scripts/useuse"
reuse Matlab
now=$(date +"%T")
echo "Current time : $now"
command="run('$ppath/code/mscript/task_"$SGE_TASK_ID"');exit;"
matlab -nodisplay -nosplash -nodesktop -r $command

echo "Finished"
now=$(date +"%T")
echo "Current time : $now"
