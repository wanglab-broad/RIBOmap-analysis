#!/bin/bash
ppath="path/to/current/folder"
outpath="$ppath/mscript"
main_mscript="$shpath/2021-11-02-Tissue-template.m"

j=1
for i in $(seq -f "%03g" 1 1004)
# cat $listpath|while read file
do
	echo "tile='Position$i'" > $outpath/task_$j".m"
	cat $main_mscript >> $outpath/task_$j".m"
	j=$((j+1))
done
