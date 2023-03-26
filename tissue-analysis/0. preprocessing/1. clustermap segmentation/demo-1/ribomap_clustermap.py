#!/usr/bin/env python
# coding: utf-8

# %% cml input
import sys, os

# tile_num = int(sys.argv[1])
# tile_num = 315
# tile = f"Position{tile_num:03}"

tile = sys.argv[1]
print(tile)

from ClusterMap.clustermap import *
from anndata import AnnData
import tifffile
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import scipy.io
from sklearn import preprocessing
import timeit

start = timeit.default_timer()

# %%
### set file folder
ppath = '/stanley/WangLab/Data/Processed/2021-11-23-Hu-MouseBrainRIBOmap/output/max_cluster/'
dpath = '/stanley/WangLab/Data/Processed/2021-11-23-Hu-MouseBrainRIBOmap/round1/'
opath = '/stanley/WangLab/Data/Processed/2021-11-23-Hu-MouseBrainRIBOmap/output/clustermap/'

if not os.path.exists(opath):
    os.mkdir(opath)
    
### set parameters
xy_radius = 50 # for knn radius (only)?
z_radius = 10
pct_filter = 0.1
dapi_grid_interval = 5
min_spot_per_cell = 5
cell_num_threshold = 0.02

# %%
### read dapi: col, row, z
dapi = tifffile.imread(os.path.join(dpath, tile, '*ch04.tif'))
dapi = np.transpose(dapi, (1,2,0))

### read spots
mat = scipy.io.loadmat(os.path.join(ppath, tile, 'goodPoints_max3d.mat'))
spots = pd.DataFrame(mat['goodSpots'])
spots.columns=['spot_location_1','spot_location_2','spot_location_3']

### convert gene_name to gene identity
gene_name = list(x[0][0] for x in mat['goodReads'])
le = preprocessing.LabelEncoder()
le.fit(gene_name)
gene = le.transform(gene_name) + 1 #minimal value of gene=1

### save genelist
genes = pd.DataFrame(le.classes_)
genes_opath = os.path.join(opath, tile)
if not os.path.exists(genes_opath):
    os.mkdir(genes_opath)
genes.to_csv(os.path.join(genes_opath, 'genelist.csv'), header=False, index=False)
spots['gene'] = gene
spots['gene'] = spots['gene'].astype('int')

### instantiate model
num_gene = np.max(spots['gene'])
gene_list = np.arange(1, num_gene+1)
num_dims = len(dapi.shape)
model = ClusterMap(spots=spots, dapi=dapi, gene_list=gene_list, num_dims=num_dims, # gauss_blur=True, sigma=8,
               xy_radius=xy_radius, z_radius=z_radius, fast_preprocess=False)

model.spots['clustermap'] = -1
model.preprocess(dapi_grid_interval=dapi_grid_interval, pct_filter=pct_filter)
model.min_spot_per_cell = min_spot_per_cell # cell_num_threshold proportion to tile size?
model.segmentation(cell_num_threshold=cell_num_threshold, dapi_grid_interval=dapi_grid_interval, add_dapi=True, use_genedis=True)

model.save_segmentation(os.path.join(opath, tile, 'spots.csv'))

### save figs 
## preprocessing
plt.figure(figsize=(10,10))
plt.imshow(dapi.max(axis=2), cmap=plt.cm.gray)
plt.scatter(model.spots.loc[model.spots['is_noise'] == 0, 'spot_location_1'], model.spots.loc[model.spots['is_noise'] == 0, 'spot_location_2'], s=0.5, c='g', alpha=.5)
plt.scatter(model.spots.loc[model.spots['is_noise'] == -1, 'spot_location_1'], model.spots.loc[model.spots['is_noise'] == -1, 'spot_location_2'], s=0.5, c='r', alpha=.5)
plt.savefig(os.path.join(opath, tile, 'spots_pp.png'))
plt.clf()
plt.close()

## segmentation 
cell_ids = model.spots['clustermap']
cells_unique = np.unique(cell_ids)
spots_repr = np.array(model.spots[['spot_location_2', 'spot_location_1']])[cell_ids>=0]
cell_ids = cell_ids[cell_ids>=0]                
cmap = np.random.rand(int(max(cell_ids)+1), 3)
fig, ax = plt.subplots(figsize=(10,10))
ax.imshow(np.zeros(dapi.max(axis=2).shape), cmap='Greys_r')
ax.scatter(spots_repr[:,1],spots_repr[:,0], c=cmap[[int(x) for x in cell_ids]], s=1, alpha=.5)
ax.scatter(model.cellcenter_unique[:,1], model.cellcenter_unique[:,0], c='r', s=3)
plt.axis('off')
plt.savefig(os.path.join(opath, tile, 'cell_seg.png'))
plt.clf()
plt.close()

fig, ax = plt.subplots(figsize=(10,10))
ax.imshow(dapi.max(axis=2), cmap='Greys_r')
ax.scatter(spots_repr[:,1],spots_repr[:,0], c=cmap[[int(x) for x in cell_ids]], s=1, alpha=.5)
ax.scatter(model.cellcenter_unique[:,1], model.cellcenter_unique[:,0], c='r', s=3)
plt.axis('off')
plt.savefig(os.path.join(opath, tile, 'cell_seg_with_dapi.png'))
plt.clf()
plt.close()

stop = timeit.default_timer()
computation_time = round((stop - start) / 60, 2)

### save log (csv)
log_dict = {'number_of_spots': spots.shape[0], 
           'number_of_spots_after_pp': model.spots.loc[model.spots['is_noise'] == 0, :].shape[0],
           'number_of_cells': model.cellcenter_unique.shape[0],
           'computation_time': computation_time}
log = pd.DataFrame(log_dict, index=[tile])
log.to_csv(os.path.join(opath, tile, 'log.csv'))