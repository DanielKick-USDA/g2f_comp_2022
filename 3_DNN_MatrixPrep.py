# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import tqdm

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)

from sklearn import preprocessing # LabelEncoder
from sklearn.metrics import mean_squared_error # if squared=False; RMSE
# -

meta = pd.read_csv('./data/Processed/meta0.csv')
# meta['Date_Planted'] = meta['Date_Planted'].astype(int)
# meta['Date_Harvested'] = meta['Date_Harvested'].astype(int)
phno = pd.read_csv('./data/Processed/phno0.csv')
soil = pd.read_csv('./data/Processed/soil0.csv')
wthr = pd.read_csv('./data/Processed/wthr0.csv')
# wthrWide = pd.read_csv('./data/Processed/wthrWide0.csv')
cgmv = pd.read_csv('./data/Processed/cgmv0.csv')

mask = ((phno.Yield_Mg_ha.notna()) | (phno.Year == 2022))
phno = phno.loc[mask, :].reset_index().drop(columns = 'index')
phno = phno.loc[:, ['Env', 'Year', 'Hybrid', 'Yield_Mg_ha']]

# # Data Prep

# ## Prep CVs

# ## Prep y

YMat = np.array(phno.Yield_Mg_ha)

# ## One Hot Encode G

temp = phno.loc[:, ['Env', 'Year', 'Hybrid', 'Yield_Mg_ha']]
temp = pd.concat([temp, temp.Hybrid.str.split('/', expand=True)], axis=1
        ).rename(columns = {0:'P0', 1:'P1'})
temp
uniq_parents = list(set(pd.concat([temp['P0'], temp['P1']])))

# +
GMat = np.zeros([temp.shape[0], len(uniq_parents)])

# for each uniq_parent 
for j in range(len(uniq_parents)):
    for parent in ['P0', 'P1']:
        mask = (temp[parent] == uniq_parents[j]) 
        GMat[temp.loc[mask, ].index, j] += 1
# -

# confirm there are two parents encoded for each observation
assert 2 == np.min(np.sum(GMat, axis = 1))

# ## Make S Matrix

SMat = phno.loc[:, ['Env']].merge(soil.drop(columns = ['Unnamed: 0', 'Year'])).drop(columns = ['Env'])
SMatNames = list(SMat)
SMat = np.array(SMat)

# ## Prep W

# +
# Input: (N,Cin,Lin)(N,Cin,Lin) or (Cin,Lin)(Cin,Lin)
# -

WMatNames = list(wthr.drop(columns = ['Unnamed: 0', 'Env', 'Year', 'Date', 'DOY']))
WMat = np.zeros([   # Pytorch uses
    phno.shape[0],  # N
    len(WMatNames), # Cin
    np.max(wthr.DOY)# Lin
])

# loop through all obs, but only add each env once (add to all relevant obs)
added_envs = []
for i in tqdm.tqdm(phno.index):
    env = phno.loc[i, 'Env']

    if env in added_envs:
        pass
    else:
        mask = (phno.Env == env)
        WMat_idxs = phno.loc[mask, ].index

        # selected data is transposed to match correct shape
        wthr_mask = (wthr.Env == env)
        WMat[WMat_idxs, :, :] = wthr.loc[wthr_mask, 
                                   ].sort_values('DOY'
                                   ).drop(columns = ['Unnamed: 0', 'Env', 
                                                     'Year', 'Date', 'DOY']).T

        added_envs += [env]

# ## Prep CGMV?

MMatNames = list(cgmv.drop(columns = ['Unnamed: 0', 'Env', 'Year']))

MMat = np.zeros([   
    phno.shape[0],  
    len(MMatNames)
])

# loop through all obs, but only add each env once (add to all relevant obs)
added_envs = []
for i in tqdm.tqdm(phno.index):
    env = phno.loc[i, 'Env']

    if env in added_envs:
        pass
    else:
        mask = (phno.Env == env)
        MMat_idxs = phno.loc[mask, ].index

        # selected data is transposed to match correct shape
        cgmv_mask = (cgmv.Env == env)
        MMat[MMat_idxs, :] = cgmv.loc[cgmv_mask, 
                                ].drop(columns = ['Unnamed: 0', 'Env', 'Year'])

        added_envs += [env]

# # Save data
# This will streamline model generation. I'll just need to load these files in and can directly begin modeling.

save_path = './data/Processed/'

if True == False:
    np.save(save_path+'GMatNames.npy', uniq_parents)
    np.save(save_path+'SMatNames.npy', SMatNames)
    np.save(save_path+'WMatNames.npy', WMatNames)
    np.save(save_path+'MMatNames.npy', MMatNames)

    phno.to_csv(save_path+'phno3.csv', index=False)

    np.save(save_path+'YMat3.npy', YMat)
    np.save(save_path+'GMat3.npy', GMat)
    np.save(save_path+'SMat3.npy', SMat)
    np.save(save_path+'WMat3.npy', WMat)
    np.save(save_path+'MMat3.npy', MMat)
