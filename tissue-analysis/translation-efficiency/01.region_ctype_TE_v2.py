import os,scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
if len(sys.argv) > 1:
    percent_cutoff = float(sys.argv[1])
else:
    print('Please input the tile number:')
    sys.exit()
import scanpy as sc
from statsmodels.stats.multitest import multipletests as fdr

sdata = sc.read_h5ad('/stanley/WangLab/Data/Analyzed/2022-09-05-Hu-Tissue/output/2022-11-11-Brain-combined-3mad-ct-v2_region_label.h5ad')
idx_t = sdata.obs['replicate'] == 'rep3'
sdata_f1 = sdata[idx_t,:].copy()
sdata_f1.X = sdata_f1.layers['raw'].copy()
sc.pp.normalize_total(sdata_f1, target_sum = 1e4)
sdata_f1.layers['norm1e4'] = sdata_f1.X.copy()
del sdata
output_path = f'/stanley/WangLab/atlas/Brain/09.TE_analysis/All_results_v4_{percent_cutoff}'
os.makedirs(output_path,exist_ok=True)

ctype_change = dict()
for i in np.unique(sdata_f1.obs['level_2']):
    ctype_change[i] = i.replace(' ','_').replace(',','_').replace('/','_').replace('__','_')
sdata_f1.obs.replace({'level_2':ctype_change},inplace=True)


region_list = ['Cerebal nuclei', 'Cortical subplate', 'Fiber tracts', 'Hippocampal region', 'Hypothalamus', 'Isocortex', 'Olfactory areas', 'Thalamus','All_region']
ctype_list = list(sdata_f1.obs['level_2'].cat.categories)
ctype_list.append('All_ctype')
# ctype_t = 'All_ctype'
for ctype_t in ctype_list:
    data_dict = dict()
    for sample_t in ['RIBOmap','STARmap']:
        if ctype_t != 'All_ctype':
            idx_sample = (sdata_f1.obs['protocol'] == sample_t) & (sdata_f1.obs['region'] != 'other') & (sdata_f1.obs['level_2'] == ctype_t)
            count_all_t = sdata_f1[idx_sample,:].obs[['z','protocol','level_2','region']].groupby(['protocol','level_2','region']).count()
            count_all_t = count_all_t.loc[sample_t,:].loc[ctype_t,:]
        else:
            idx_sample = (sdata_f1.obs['protocol'] == sample_t) & (sdata_f1.obs['region'] != 'other')
            count_all_t = sdata_f1[idx_sample,:].obs[['z','protocol','region']].groupby(['protocol','region']).count()
            count_all_t = count_all_t.loc[sample_t,:]
        count_all_t.loc['All_region'] = count_all_t['z'].sum()
        count_all_t = count_all_t.loc[region_list,:].copy()

        data_t = pd.DataFrame(sdata_f1[idx_sample,:].layers[f'norm1e4'].copy(),columns=sdata_f1.var.index.values)
        count_t = data_t.copy()
        print(f'{sample_t}: {data_t.shape[0]} Cells, {np.sum(idx_sample)} Cells')
        ### mean values
        mean_region_all_t = data_t.mean(0)
        data_t['region'] = np.array(sdata_f1[idx_sample,:].obs[['region']])
        mean_region = data_t.groupby('region').mean()
        mean_region.loc['All_region',:] = mean_region_all_t.copy()
        mean_region = mean_region.loc[region_list,:]
        ### valid cell counts
        count_t[count_t>0] = 1
        validcount_region_all_t = count_t.sum(0)
        count_t['region'] = np.array(sdata_f1[idx_sample,:].obs[['region']])
        validcount_region = count_t.groupby('region').sum()
        validcount_region.loc['All_region',:] = validcount_region_all_t.copy()
        validcount_region = validcount_region.loc[region_list,:]
        
        cell_percentage = np.array(validcount_region)/np.array(count_all_t)
        mean_region_t = np.array(mean_region)
        mean_region_t[cell_percentage < percent_cutoff] = np.nan
        mean_region.loc[:] = mean_region_t
        ### save
        data_dict[f'{ctype_t}_{sample_t}_cell_percentage'] = pd.DataFrame(cell_percentage,columns=mean_region.columns,index=mean_region.index)
        data_dict[f'{ctype_t}_{sample_t}_mean'] = mean_region
        data_dict[f'{ctype_t}_{sample_t}_validcount'] = validcount_region
    ### TE analysis
    te_t = data_dict[f'{ctype_t}_RIBOmap_mean'] / data_dict[f'{ctype_t}_STARmap_mean']
    te_t[np.isinf(te_t)] = np.nan
    data_dict[f'{ctype_t}_TE_Raw'] = te_t
    te_t = np.log2(te_t)
    te_t[np.isinf(te_t)] = np.nan
    te_t_good_zscore = scipy.stats.zscore(te_t,axis = 0,nan_policy = 'omit')
    data_dict[f'{ctype_t}_TE'] = te_t_good_zscore
    ### output examples
    for i in data_dict.keys():
        data_dict[i].to_csv(f'{output_path}/{i}.csv')
    del data_dict

# #### region vs ctype 
# region_list = ['Cerebal nuclei', 'Cortical subplate', 'Fiber tracts', 'Hippocampal region', 'Hypothalamus', 'Isocortex', 'Olfactory areas', 'Thalamus','All_region']
# # region_list = ['Cerebal nuclei']
# ctype_list = list(sdata_f1.obs['level_2'].cat.categories)
# ctype_list.append('All_ctype')
# # ctype_t = 'All_ctype'
for region_t in region_list:
    data_dict = dict()
    for sample_t in ['RIBOmap','STARmap']:
        if region_t != 'All_region':
            idx_sample = (sdata_f1.obs['protocol'] == sample_t) & (sdata_f1.obs['region'] != 'other') & (sdata_f1.obs['region'] == region_t)
            count_all_t = sdata_f1[idx_sample,:].obs[['z','protocol','region','level_2']].groupby(['protocol','region','level_2']).count()
            count_all_t = count_all_t.loc[sample_t,:].loc[region_t,:]
        else:
            idx_sample = (sdata_f1.obs['protocol'] == sample_t) & (sdata_f1.obs['region'] != 'other')
            count_all_t = sdata_f1[idx_sample,:].obs[['z','protocol','level_2']].groupby(['protocol','level_2']).count()
            count_all_t = count_all_t.loc[sample_t,:]
        count_all_t.loc['All_ctype'] = count_all_t['z'].sum()
        count_all_t = count_all_t.loc[ctype_list,:].copy()

        data_t = pd.DataFrame(sdata_f1[idx_sample,:].layers[f'norm1e4'].copy(),columns=sdata_f1.var.index.values)
        count_t = data_t.copy()
        print(f'{sample_t}: {data_t.shape[0]} Cells, {np.sum(idx_sample)} Cells')
        ### mean values
        mean_ctype_all_t = data_t.mean(0)
        data_t['level_2'] = np.array(sdata_f1[idx_sample,:].obs[['level_2']])
        mean_ctype = data_t.groupby('level_2').mean()
        mean_ctype.loc['All_ctype',:] = mean_ctype_all_t.copy()
        mean_ctype = mean_ctype.loc[ctype_list,:]
        ### valid cell counts
        count_t[count_t>0] = 1
        validcount_ctype_all_t = count_t.sum(0)
        count_t['level_2'] = np.array(sdata_f1[idx_sample,:].obs[['level_2']])
        validcount_ctype = count_t.groupby('level_2').sum()
        validcount_ctype.loc['All_ctype',:] = validcount_region_all_t.copy()
        validcount_ctype = validcount_ctype.loc[ctype_list,:]


        cell_percentage = np.array(validcount_ctype)/np.array(count_all_t)
        mean_ctype_t = np.array(mean_ctype)
        mean_ctype_t[cell_percentage < percent_cutoff] = np.nan
        mean_ctype.loc[:] = mean_ctype_t
        ### save
        data_dict[f'{region_t}_{sample_t}_cell_percentage'] = pd.DataFrame(cell_percentage,columns=mean_ctype.columns,index=mean_ctype.index)
        data_dict[f'{region_t}_{sample_t}_mean'] = mean_ctype
        data_dict[f'{region_t}_{sample_t}_validcount'] = validcount_ctype
    ### TE analysis
    te_t = data_dict[f'{region_t}_RIBOmap_mean'] / data_dict[f'{region_t}_STARmap_mean']
    te_t[np.isinf(te_t)] = np.nan
    data_dict[f'{region_t}_TE_Raw'] = te_t
    te_t = np.log2(te_t)
    te_t[np.isinf(te_t)] = np.nan
    te_t_good_zscore = scipy.stats.zscore(te_t,axis = 0,nan_policy = 'omit')
    data_dict[f'{region_t}_TE'] = te_t_good_zscore
    ### output examples
    for i in data_dict.keys():
        data_dict[i].to_csv(f'{output_path}/{i}.csv')
    del data_dict
