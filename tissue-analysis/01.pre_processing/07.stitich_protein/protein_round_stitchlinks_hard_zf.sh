#!/bin/bash -l
# Generate hard stitchlinks for DAPI stitching on workstation

protein=$1 ## NeuN
sample=$2 ## Brain_RIBO
tissue=`echo $sample | cut -f1 -d '_'`
tech=`echo $sample | cut -f2 -d '_'`

outpath="/stanley/WangLab/atlas/Brain/10.protein_stitch/01.stitchlinks/$sample/$protein""_hard_link"
mkdir -p $outpath
cp /stanley/WangLab/atlas/Brain/04.stitched_results/stitchlinks/$sample/TileConfiguration.registered.txt $outpath/
#blanktilepath="../../../../00.key_files/blank.tif"
orderlist="/stanley/WangLab/atlas/$tissue/00.sh/00.orderlist/orderlist_$sample.csv"

i=1
for j in `cat $orderlist`
do 
  j=`printf "%03d" $j`
	#file_t="/stanley/WangLab/Data/Processed/2022-09-05-Hu-Tissue"$tech"map/output/registered_images/"$protein"/Position"$j".tif"
	file_t="/stanley/WangLab/Data/Processed/2022-09-05-Hu-Tissue"$tech"map/output/registered_images_2d/"$protein"/Position"$j".tif"
  if [ "$j" = "0" ] || [ ! -s $file_t ] 
  then
		echo "ln /stanley/WangLab/atlas/00.key_files/blank_zf.tif $outpath/tile_$i.tif"
##    ln /stanley/WangLab/atlas/Brain/00.sh/07.stitich_protein/blank.tif $outpath/tile_$i.tif
#    ln /stanley/WangLab/atlas/00.key_files/blank_zf.tif $outpath/tile_$i.tif
  else
		echo "ln $file_t $outpath/tile_$i.tif "
    #ln $file_t $outpath/tile_$i.tif 
  fi
  i=$((i+1))
done
