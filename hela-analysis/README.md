# HeLa cell culture data analysis

0. ```preprocessing```
	* ```0. amplicon identification demo.mlx```
	* ```1. staining image registration demo```
	* ```2. image stitching and segmentation demo.mlx```

1. ```reads assignment fucci.ipynb```: contains code for assigning amplicons to cells based on 3D cell segmentation.
2. ```generate h5ad input.ipynb```: generate scanpy h5ad file for downstream analysis
3. ```preprocessing and filtering.ipynb```: single cell quanlity assessment and filtering.
4. ```generate cell cycle label.ipynb```: generate gene expresion based cell cycle labels and compare with FUCCI signal.
5. ```co-expression.ipynb```: contains code for gene co-variation analysis.

6. ```co-localization.ipynb```: contains code for gene co-localization analysis.

7. ```colocalization_permutation.ipynb```

8. ```spatial_funcs.py```: contains functions related to gene co-localization analysis.

9. ```cell_cycle_markers.xlsx```: annotated cell cycle gene markers.
