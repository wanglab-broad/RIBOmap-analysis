qsub -l h_vmem=50G -t 1-2 run_python.sh 01.region_ctype_TE_v3.py plist_v3
qsub -l h_vmem=50G -t 1-5 run_python.sh 02.region_ctype_TE_RIBOmap_rep2_v1.py 02.plist_v1
