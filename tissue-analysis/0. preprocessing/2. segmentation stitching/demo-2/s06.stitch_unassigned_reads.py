import os, sys, copy, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance
from tqdm import tqdm
from glob import glob
import matplotlib
from itertools import chain

# Parameters
sample_id = sys.argv[1]
suffix = sys.argv[2] # '_tb1'
t = sys.argv[3]
tissue = sample_id.split('_')[0]
start_t = 1
alignment_thresh = 0.5


############################# FUNCTIONS #############################

# Read in FIJI Tile Coordinates (TileConfiguration.registered.txt and TileConfiguration.txt)
def get_coords(path, grid=False):
    f = open(path)
    line = f.readline()
    list = []
    while line:
        if line.startswith('tile'):
            a = np.array(line.replace('tile_','').replace('.tif; ; (',',').replace(', ',',').replace(')\n','').split(','))
            if not grid:
                a = [math.floor(float(x)+0.5) for x in a] # for proper rounding
            else:
                a = (a.astype(float)+0.5).astype(int).tolist()
                a[1:3] = (np.divide(a[1:3],int(img_c*0.9+0.5)) + 0.5).astype(int).tolist()
            list.append(a)
        line = f.readline()
    coords_df = np.array(list)
    f.close
    return coords_df
# Find closest cell center for multiassigned reads
def closest_node(node, nodes):
    closest_index = distance.cdist([node], nodes).argmin()
    return nodes[closest_index], closest_index

def idxs_within_coords(df, top_edge, bottom_edge, right_edge, left_edge, include=True):
    if include:
        idxs = df[(df['column'] <= right_edge) & (df['column'] >= left_edge) & (df['row'] >= top_edge) & (df['row'] <= bottom_edge)]['cell_barcode'].tolist()
    else:
        idxs = df[(df['column'] < right_edge) & (df['column'] > left_edge) & (df['row'] > top_edge) & (df['row'] < bottom_edge)]['cell_barcode'].tolist()
    return idxs

def idxs_within_coords_reads(df, top_edge, bottom_edge, right_edge, left_edge, include=True):
    if include:
        idxs = df[(df['column'] <= right_edge) & (df['column'] >= left_edge) & (df['row'] >= top_edge) & (df['row'] <= bottom_edge)]['reads_barcode'].tolist()
    else:
        idxs = df[(df['column'] < right_edge) & (df['column'] > left_edge) & (df['row'] > top_edge) & (df['row'] < bottom_edge)]['reads_barcode'].tolist()
    return idxs


# Function for filtering multi-assigned reads, takes in a pandas grouping object
def filter_multi_assign(grouping):
    """
    Calculates closest cell for a set of identical multi-assigned reads for a pandas groupby element.
    Returns: {id: cell_barcode} (dict): a dictionary of the duplicated id and list of barcodes that need to be filtered out (i.e. non-selected)
    """
    (id, df) = grouping
    xyz = id.split('-')[0:3] # xyz coord of multi-assigned reads
    # identify cells that reads are assigned to
    repeat_reads_cell_index = cell_center['cell_barcode'].isin(df['cell_barcode'])
    repeat_reads_cell = cell_center.loc[repeat_reads_cell_index,:]
    # calculate closest cell (according to cell center)
    closest_index = closest_node(xyz,np.array(repeat_reads_cell.loc[:,['column', 'row', 'z_axis']]).tolist())[1]
    selected_cell = repeat_reads_cell.iloc[closest_index,0] # barcode of closest cell
    # get indices of other reads to be filtered from farther cells
    filtered_read_idxs = df.index[np.logical_not(remain_reads.loc[df.index,'cell_barcode'] == selected_cell)].tolist()
    return({id:filtered_read_idxs})

# Get figsize according to image size
def get_figsize(coords_df_tuned, scale=5):
    max_col_coord = coords_df_tuned['column_coord_obs'].max()
    max_row_coord = coords_df_tuned['row_coord_obs'].max()
    return([max_col_coord/100/scale * 2, max_row_coord/100/scale * 2])

#####################################################################


print(f"Starting full sample stitch for {sample_id}")

# Set variables and file i/o
img_c, img_r = [2048, 2048]
data_dir = f'/stanley/WangLab/atlas/{tissue}'
orderlist = glob(os.path.join(data_dir, '00.sh', '00.orderlist', f'orderlist_{sample_id}.csv'))[0]
inputpath = os.path.join(data_dir, '04.stitched_results', 'stitchlinks', sample_id)
readspath = os.path.join(data_dir, '03.segmentation', sample_id)
outputpath = os.path.join(data_dir, '04.stitched_results', sample_id)
os.makedirs(outputpath, exist_ok=True)

cell_barcode_min = 0
middle_edge = 0
remain_reads = pd.DataFrame({'gene_name':[],'spot_location_1':[],'spot_location_2':[],'spot_location_3':[],'gene':[],'is_noise':[],'cell_barcode':[],'raw_cell_barcode':[],'gridc_gridr_tilenum':[]})
cell_center = pd.DataFrame({'cell_barcode':[],'column':[],'row':[],'z_axis':[],'raw_cell_barcode':[],'gridc_gridr_tilenum':[]})

#### Read in tile coordinates and orderlist
obs_coords = get_coords(os.path.join(inputpath, 'TileConfiguration.registered.txt'))
exp_coords = get_coords(os.path.join(inputpath, 'TileConfiguration.txt'))
grid_order = get_coords(os.path.join(inputpath, 'TileConfiguration.txt'), grid=True)

## order list
order_df = pd.read_csv(orderlist, header = None, dtype=str)
order_df.index = range(1,order_df.shape[0]+1)
order_df.columns = ['tile']

# Add expected grid configuration to observed coordinates
print("Combining read configurations...")
# Format into dataframes
coords_df = pd.DataFrame(obs_coords, columns=['index','column_coord_obs','row_coord_obs'], index = obs_coords[:,0])
coords_exp_df = pd.DataFrame(exp_coords, columns=['index','column_coord_exp','row_coord_exp'], index = exp_coords[:,0])
grid_df = pd.DataFrame(grid_order, columns=['index', 'column_count', 'row_count'], index = grid_order[:,0])
# Merge 
coords_df = coords_df[['column_coord_obs', 'row_coord_obs']].merge(coords_exp_df[['column_coord_exp', 'row_coord_exp']], right_index=True, left_index=True)
coords_df = coords_df.merge(grid_df[['column_count', 'row_count']], right_index=True, left_index=True)
# # Sort index and add corresponding tile information
coords_df = coords_df.loc[range(1,coords_df.shape[0] + 1),:]
coords_df['tile'] = order_df['tile']
#save
# coords_df.to_csv(os.path.join(outputpath,'coords.csv'))



# Zero-center (tune) registered coordinates
print("Tuning coordinates...")
## find origin and center the coords
coords_df_without_blank = coords_df.loc[coords_df['tile'].astype(int) > 0,:]
min_column, min_row = [np.min(coords_df_without_blank['column_coord_obs']), np.min(coords_df_without_blank['row_coord_obs'])]
max_column, max_row = [np.max(coords_df_without_blank['column_coord_obs']), np.max(coords_df_without_blank['row_coord_obs'])]
shape_column, shape_row = [max_column - min_column + img_c, max_row - min_row + img_r]
coords_df_tuned = copy.deepcopy(coords_df)
coords_df_tuned['column_coord_obs'] = coords_df['column_coord_obs'] - min_column
coords_df_tuned['row_coord_obs'] = coords_df['row_coord_obs'] - min_row
grid_c,grid_r = (np.max(coords_df_tuned.loc[:,['column_count','row_count']]) + 1).tolist()
# save
coords_df_tuned['tile'] = coords_df_tuned['tile'].astype(int)
# coords_df_tuned.to_csv(os.path.join(outputpath,'tuned_coords.csv'))



# ###################################################### unassigned reads (jiahao version) #######################################################
# ### STITCH
# print("============STITCHING============")
# cell_barcode_min = 0
# # remain_reads = pd.DataFrame({'gene_name':[],'spot_location_1':[],'spot_location_2':[],'spot_location_3':[],'gene':[],'is_noise':[],'cell_barcode':[],'raw_cell_barcode':[],'gridc_gridr_tilenum':[]})
# remain_reads = []

# for t_grid_c in range(0,grid_c):
# # for t_grid_c in range(0,2):
# # origin [0,0] at upper left of grid
#     # Get median column coordinate for approx alignment
#     median_col_coord = np.median(coords_df_tuned[(coords_df_tuned.column_count == t_grid_c) & (coords_df_tuned.tile != 0)]['column_coord_obs'])

#     for t_grid_r in range(0,grid_r):
#     # for t_grid_r in range(0,2):
#         # Get median row coordinate for approx alignment
#         median_row_coord = np.median(coords_df_tuned[(coords_df_tuned.row_count == t_grid_r) & (coords_df_tuned.tile != 0)]['row_coord_obs'])
#         top_middle_edge = None
#         left_middle_edge = None

#         # Get tile coordinate and grid order information 
#         order = coords_df_tuned.index[(coords_df_tuned['column_count']==t_grid_c) & (coords_df_tuned['row_count']==t_grid_r)][0]
#         tilenum = coords_df_tuned['tile'][order] # Tile number (blanks = 0)
#         if tilenum == 0: # skip blanks
#             continue
#         upper_left = coords_df_tuned.loc[order, ['column_coord_obs', 'row_coord_obs']] # upper left coordinate of tile
#         upper_left_new = copy.deepcopy(upper_left)

#         # If either column or row alignment shifted more than 1.5x tile width/height away from median coordinate, throw out
#         left_thresh = median_col_coord - (1+alignment_thresh)*img_c
#         right_thresh = median_col_coord + (1+alignment_thresh)*img_c
#         upper_thresh = median_row_coord - (1+alignment_thresh)*img_r
#         lower_thresh = median_row_coord + (1+alignment_thresh)*img_r
#         if upper_left[0] >= right_thresh or upper_left[0] <= left_thresh or upper_left[1] >= lower_thresh or upper_left[1] <= upper_thresh:
#             print(f"- Tile {tilenum} is aligned too far away from its expected position.")
#             print(f"\tTile coord: [{upper_left[0]}, {upper_left[1]}]. Median coord: [{median_col_coord}, {median_row_coord}]")
#             continue

#         # Get tile read and cell center information
#         dfpath = readspath + '/Position' + str(tilenum).zfill(3)
#         if os.path.exists(os.path.join(dfpath, 'remain_reads.csv')):
#             print(f"- Tile {tilenum}: Reads file empty. [{t_grid_c},{t_grid_r}]")
#             continue
#         if not os.path.exists(os.path.join(dfpath,'remain_reads' + suffix + '.csv')):
#             print(f"- Tile {tilenum}: Reads file does not exist. [{t_grid_c},{t_grid_r}]")
#             continue
#         remain_reads_t = pd.read_csv(os.path.join(dfpath,'remain_reads_raw' + suffix + '.csv'),index_col=0)
#         remain_reads_t = remain_reads_t.loc[remain_reads_t['clustermap'] == -1, :]
#         # cell_center_t = pd.read_csv(os.path.join(dfpath,'cell_center' + suffix + '.csv'),index_col=0)
#         if remain_reads_t.shape[0] == 0:
#             print(f"- Tile {tilenum}: Reads file empty. [{t_grid_c},{t_grid_r}]")
#             continue
#         else:
#             print(f"+ Tile number: {tilenum} | Number of reads: {remain_reads_t.shape[0]} ")

#         # Format reads data and add in grid/order information
#         remain_reads_t.rename(columns = {'clustermap':'cell_barcode'}, inplace = True)
#         remain_reads_t['gridc_gridr_tilenum'] = str(t_grid_c)+","+str(t_grid_r)+","+str(tilenum)

#         # Adjust read coordinates and cell center coordinates using observed upper left tile coordinate
#         remain_reads_t['spot_location_1'] = remain_reads_t['spot_location_1'] + upper_left[0]# col
#         remain_reads_t['spot_location_2'] = remain_reads_t['spot_location_2'] + upper_left[1]# col

#         remain_reads.append(remain_reads_t)
#         # remain_reads.reset_index(inplace=True, drop=True)

# print(f"\tTotal Reads: {len(remain_reads)}")


# remain_reads = pd.concat(remain_reads)

# plt.figure(figsize=get_figsize(coords_df_tuned,scale=10))
# plt.scatter(remain_reads['spot_location_1'], remain_reads['spot_location_2'], s=1)
# plt.gca().invert_yaxis()
# plt.savefig(os.path.join(outputpath,'unassigned_reads_' + str(t) + '.png'))

# remain_reads.rename(columns = {'spot_location_1':'column', 'spot_location_2':'row','spot_location_3':'z'}, inplace = True)
# remain_reads.to_csv(os.path.join(outputpath,'unassigned_reads_' + str(t) + '.csv'))




############################################# unassigned reads (adjust overlap region) ##########################################
### STITCH
print("============STITCHING============")
reads_barcode_min = 0
remain_reads = pd.DataFrame({'gene_name':[],'column':[],'row':[],'z':[],'gene':[],'is_noise':[],'cell_barcode':[],'gridc_gridr_tilenum':[],'reads_barcode':[],'raw_reads_barcode':[]})

for t_grid_c in range(0,grid_c):
# for t_grid_c in range(0,2):
# origin [0,0] at upper left of grid
    # Get median column coordinate for approx alignment
    median_col_coord = np.median(coords_df_tuned[(coords_df_tuned.column_count == t_grid_c) & (coords_df_tuned.tile != 0)]['column_coord_obs'])

    for t_grid_r in range(0,grid_r):
    # for t_grid_r in range(0,2):
        # Get median row coordinate for approx alignment
        median_row_coord = np.median(coords_df_tuned[(coords_df_tuned.row_count == t_grid_r) & (coords_df_tuned.tile != 0)]['row_coord_obs'])
        top_middle_edge = None
        left_middle_edge = None

        # Get tile coordinate and grid order information 
        order = coords_df_tuned.index[(coords_df_tuned['column_count']==t_grid_c) & (coords_df_tuned['row_count']==t_grid_r)][0]
        tilenum = coords_df_tuned['tile'][order] # Tile number (blanks = 0)
        if tilenum == 0: # skip blanks
            continue
        upper_left = coords_df_tuned.loc[order, ['column_coord_obs', 'row_coord_obs']] # upper left coordinate of tile
        upper_left_new = copy.deepcopy(upper_left)

        # If either column or row alignment shifted more than 1.5x tile width/height away from median coordinate, throw out
        left_thresh = median_col_coord - (1+alignment_thresh)*img_c
        right_thresh = median_col_coord + (1+alignment_thresh)*img_c
        upper_thresh = median_row_coord - (1+alignment_thresh)*img_r
        lower_thresh = median_row_coord + (1+alignment_thresh)*img_r
        if upper_left[0] >= right_thresh or upper_left[0] <= left_thresh or upper_left[1] >= lower_thresh or upper_left[1] <= upper_thresh:
            print(f"- Tile {tilenum} is aligned too far away from its expected position.")
            print(f"\tTile coord: [{upper_left[0]}, {upper_left[1]}]. Median coord: [{median_col_coord}, {median_row_coord}]")
            continue

        # Get tile read and cell center information
        dfpath = readspath + '/Position' + str(tilenum).zfill(3)
        if os.path.exists(os.path.join(dfpath, 'remain_reads.csv')):
            print(f"- Tile {tilenum}: Reads file empty. [{t_grid_c},{t_grid_r}]")
            continue
        if not os.path.exists(os.path.join(dfpath,'remain_reads' + suffix + '.csv')):
            print(f"- Tile {tilenum}: Reads file does not exist. [{t_grid_c},{t_grid_r}]")
            continue
        remain_reads_t = pd.read_csv(os.path.join(dfpath,'remain_reads_raw' + suffix + '.csv'),index_col=0)
        remain_reads_t = remain_reads_t.loc[remain_reads_t['clustermap'] == -1, :]
        if remain_reads_t.shape[0] == 0:
            print(f"- Tile {tilenum}: Reads file empty. [{t_grid_c},{t_grid_r}]")
            continue
        else:
            print(f"+ Tile number: {tilenum} | Number of reads: {remain_reads_t.shape[0]} ")

        # Format reads data and add in grid/order information
        remain_reads_t.rename(columns = {'clustermap':'cell_barcode','spot_location_1':'column', 'spot_location_2':'row','spot_location_3':'z'}, inplace = True)
        remain_reads_t['gridc_gridr_tilenum'] = str(t_grid_c)+","+str(t_grid_r)+","+str(tilenum)

        # Adjust read coordinates and cell center coordinates using observed upper left tile coordinate
        remain_reads_t['column'] = remain_reads_t['column'] + upper_left[0]# col
        remain_reads_t['row'] = remain_reads_t['row'] + upper_left[1]# col
        remain_reads_t['reads_barcode'] = remain_reads_t.index.values
        remain_reads_t['raw_reads_barcode'] = remain_reads_t['reads_barcode']
        remain_reads_t['reads_barcode'] = remain_reads_t['reads_barcode'] + reads_barcode_min + 1

        # Evaluate previous neighbors (i.e. tiles directly left and up from current tile)
        t_grid_c_previous = t_grid_c - 1
        t_grid_r_previous = t_grid_r - 1
        remain_reads_drop_barcodes = []
        remain_reads_t_drop_barcodes = []

        # Get middle edge with left tile if it isn't blank/doesn't exist
        if t_grid_c_previous >= 0: # current tile not on left edge (first column)
            order_left = coords_df_tuned.index[(coords_df_tuned['column_count']==t_grid_c_previous) & (coords_df_tuned['row_count']==t_grid_r)][0]
            #print(f'Left tile: {order_left}')
            if coords_df_tuned.loc[order_left,'tile'] != 0: # check left tile not blank
                # Calculate overlap region between new tile and left tile

                upper_left_left = coords_df_tuned.loc[order_left, ['column_coord_obs', 'row_coord_obs']]
                new_lower = upper_left[1] > upper_left_left[1] # if equal, then doesn't matter which to coord to use from the two
                left_overlap_top = upper_left[1] # row coord
                left_overlap_bot = upper_left_left[1] + img_r if new_lower else upper_left[1] + img_r # row coord
                left_overlap_left = upper_left[0] # col coord
                left_overlap_right = upper_left_left[0] + img_c # col coord
                left_middle_edge = left_overlap_left + (left_overlap_right - left_overlap_left) / 2

                # Identify cells to remove based on split overlap region to avoid duplicates
                if left_middle_edge >= left_overlap_left and left_overlap_right - left_overlap_left <= img_c and left_overlap_bot - left_overlap_top <= img_r:
                    # Old region cells (remove right half of overlap) 
                    remain_reads_drop_barcodes += idxs_within_coords_reads(remain_reads, left_overlap_top, left_overlap_bot, left_overlap_right, left_middle_edge, include=False)
                    # New tile cells (remove left half of overlap)
                    remain_reads_t_drop_barcodes += idxs_within_coords_reads(remain_reads_t, left_overlap_top, left_overlap_bot, left_middle_edge, left_overlap_left, include=False)
                else:
                    print(f"!!! Warning: no overlap region between tile {tilenum} and left region !!!")
                    
        # Get middle edge with top tile if it isn't blank/doesn't exist
        if t_grid_r_previous >= 0: # current tile not on top edge (first row)
            order_top = coords_df_tuned.index[(coords_df_tuned['column_count']==t_grid_c) & (coords_df_tuned['row_count']==t_grid_r_previous)][0]
            #print(f'Top tile: {order_top}')
            if coords_df_tuned.loc[order_top,'tile'] != 0: # check top tile not blank
                # Calculate overlap region between new tile and top tile
                upper_left_top = coords_df_tuned.loc[order_top, ['column_coord_obs', 'row_coord_obs']]
                new_right = upper_left[0] > upper_left_top[0] # if equal, then doesn't matter which to coord to use from the two
                top_overlap_top = upper_left[1] # row coord
                top_overlap_bot = upper_left_top[1] + img_r # row coord
                top_overlap_left = upper_left[0] # col coord
                top_overlap_right = upper_left_top[0] + img_c if new_right else upper_left[0] + img_c # col coord
                top_middle_edge = top_overlap_top + (top_overlap_bot - top_overlap_top) / 2
                # Identify cells to remove based on split overlap region to avoid duplicates
                if top_middle_edge >= top_overlap_top and top_overlap_right - top_overlap_left <= img_c and top_overlap_bot - top_overlap_top <= img_r:
                    # Old region cells (remove lower half of overlap) 
                    remain_reads_drop_barcodes += idxs_within_coords_reads(remain_reads, top_middle_edge, top_overlap_bot, top_overlap_right, top_overlap_left, include=True)
                    # New tile cells (remove upper half of overlap)
                    remain_reads_t_drop_barcodes += idxs_within_coords_reads(remain_reads_t, top_overlap_top, top_middle_edge, top_overlap_right, top_overlap_left, include=True)
                else:
                    print(f"!!! Warning: no overlap region between tile {tilenum} and top region !!!")
        rescued_barcodes = []
        for i in np.unique(remain_reads_drop_barcodes):
            row_coord, col_coord = remain_reads.loc[remain_reads['reads_barcode']==i, ['row', 'column']].values[0]
            if top_middle_edge:
                # Remove cell_barcodes from those to drop above top_middle_edge from the left tile
                if row_coord < top_middle_edge:
                    rescued_barcodes.append(i)
            if left_middle_edge:
                if col_coord < left_middle_edge:
                    rescued_barcodes.append(i)
        remain_reads_drop_barcodes = [idx for idx in remain_reads_drop_barcodes if idx not in rescued_barcodes]

        # Drop cells and associated reads
        remain_reads_drop_barcodes = np.unique(remain_reads_drop_barcodes)
        remain_reads_t_drop_barcodes = np.unique(remain_reads_t_drop_barcodes)
        previous_cell_count = len(remain_reads)
        matched_cell_count = len(remain_reads[remain_reads['reads_barcode'].isin(remain_reads_drop_barcodes)].index)

        if len(remain_reads_drop_barcodes) != matched_cell_count:
            print(f"================== WARNING! Number of unique barcodes to drop: {len(remain_reads_drop_barcodes)} != Number of barcodes that match in remain_reads: {matched_cell_count}")
        #remain_reads = remain_reads[~remain_reads['reads_barcode'].isin(remain_reads_drop_barcodes)] # using this method because indices are not unique
        remain_reads.drop(remain_reads[remain_reads['reads_barcode'].isin(remain_reads_drop_barcodes)].index, inplace=True)
        if len(remain_reads_drop_barcodes) != previous_cell_count - len(remain_reads):
            print(f'{len(remain_reads_drop_barcodes)} cells should have been dropped from old region (reality: {previous_cell_count - len(remain_reads)})')
        remain_reads_t.drop(remain_reads_t[remain_reads_t['reads_barcode'].isin(remain_reads_t_drop_barcodes)].index, inplace=True)

        if remain_reads_t.shape[0] > 0:
            reads_barcode_min = np.max(remain_reads_t['reads_barcode']) + 1


        remain_reads = remain_reads.append(remain_reads_t)
        remain_reads.reset_index(inplace=True, drop=True)

print(f"\tTotal Reads: {remain_reads.shape[0]}")


plt.figure(figsize=get_figsize(coords_df_tuned,scale=10))
plt.scatter(remain_reads['column'], remain_reads['row'], s=1)
plt.gca().invert_yaxis()
plt.savefig(os.path.join(outputpath,'unassigned_reads_adjust_overlap_' + str(t) + '.png'))
remain_reads_f1 = remain_reads.loc[remain_reads['is_noise'] != -1,:].copy()
plt.figure(figsize=get_figsize(coords_df_tuned,scale=10))
plt.scatter(remain_reads_f1['column'], remain_reads_f1['row'], s=1)
plt.gca().invert_yaxis()
plt.savefig(os.path.join(outputpath,'unassigned_reads_adjust_overlap_no_noise_' + str(t) + '.png'))
remain_reads.to_csv(os.path.join(outputpath,'unassigned_reads_adjust_overlap_' + str(t) + '.csv'))
