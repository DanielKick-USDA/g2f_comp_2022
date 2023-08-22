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
import os, re
import numpy as np
import pandas as pd

import sklearn
from sklearn.linear_model import LinearRegression # for solving for scaling /centering values
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error # if squared=False; RMSE

import os
import optuna
import pickle as pkl

import tqdm

from joblib import Parallel, delayed # Oputna has parallelism built in but for training replicates of the selected model
# I'll run them through Parallel

import plotly.express as px
import plotly.graph_objects as go

# +
cache_path = './notebook_artifacts/5_ensembling/'

shared_out_path = './data/Shared_Model_Output'
# -

# # Level 1 Modeling: ID prediction files

# ## Blup Predictions

blup_yhats = [e for e in os.listdir(shared_out_path) if re.findall('.+BlupYHats.csv', e)]
blup_yhats = [e for e in blup_yhats if re.findall('\d+', e)]

# +
# task 1. For each BlupYHat file, get the corresponding phno_ref 

file_name = 'Ws20163BlupYHats.csv'

def load_blup_yhats(file_name):
    df = pd.read_csv(shared_out_path + '/'+ file_name
                    ).rename(columns = {'Unnamed: 0': 'Ind',
                                       'Yield_Mg_ha': 'Yield_Mg_ha_Scaled',
                                       'YHat': 'YHat_Scaled'})
    blup_yhats_dig = re.findall('\d+', file_name)[0]    
    df['Class'] = re.findall('\D+', file_name)[0]   
    df['CV'] = blup_yhats_dig[0:4]
    df['Rep'] = blup_yhats_dig[-1]
    ref = pd.read_csv('./data/Processed/phno_ref_'+str(blup_yhats_dig[0:4])+'_small.csv'
               ).rename(columns = {'Unnamed: 0': 'Ind'})

    df = ref.merge(df, how = 'outer')
    
    # solve for the scaling /centering values
    vals = df.loc[:, ['Yield_Mg_ha', 'Yield_Mg_ha_Scaled']].dropna()
    val_x = vals[['Yield_Mg_ha_Scaled']]
    val_y = vals[['Yield_Mg_ha']]

    reg = LinearRegression().fit(val_x, val_y)
    center_val = reg.intercept_[0]
    scale_val = reg.coef_[0][0]
    
    df['Center'] = center_val
    df['Scale'] = scale_val
    
    df['YHat'] = (df['YHat_Scaled']*scale_val) + center_val

    return(df)

df = load_blup_yhats(file_name = 'Ws20163BlupYHats.csv')
df


# -

def calc_blup_rmses(df):
    """
    This code assumes columns 'Year', 'Yield_Mg_ha', 'YHat'
    """
    def blup_groupby_rmse(df):
        #https://stackoverflow.com/questions/47914428/python-dataframe-calculating-r2-and-rmse-using-groupby-on-one-column
        return(sklearn.metrics.mean_squared_error(df['Yield_Mg_ha'], df['YHat'], squared = False))

    df_rmses = df.loc[:, ['Year', 'Yield_Mg_ha', 'YHat']].dropna().groupby('Year'
                ).apply(blup_groupby_rmse).reset_index().rename(columns = {0:'RMSE'})
    df_rmses = df_rmses.merge(df.loc[:, ['Class', 'CV', 'Rep']], right_index=True, left_index = True)
    return(df_rmses)


calc_blup_rmses(df)

blup_rmse_overview = pd.concat([calc_blup_rmses(df) for df in [load_blup_yhats(e) for e in blup_yhats]])
blup_rmse_overview

# ## Random Forests

rf_yhats = [e for e in os.listdir(shared_out_path) if re.findall('rf\d\d\d\d_\drfYHats.csv', e)]

# ## XGBoost

xgb_yhats = [e for e in os.listdir(shared_out_path) if re.findall('xgb\d\d\d\d_\dxgbYHats.csv', e)]

# ## DNNs

# +
# Placeholder
# -

# ## Historical

hist_yhats = [e for e in os.listdir(shared_out_path) if re.findall('hist\d\d\d\d_\dYHats.csv', e)]

# # Level 2 Modeling: Collect Predictions

# ## No BLUP version 

# data for merging predictions into
phno = pd.read_csv('./data/Processed/phno3.csv')
phno.head()


# ### Identify Files

# from a list get model 
def parse_yhat_filename(in_str = 'rf2017_5rfYHats.csv'):
    in_str = re.findall('^\D+\d+_\d', in_str)[0]
    mod = re.findall('^\D+', in_str)[0]
    cv, rep = in_str.replace(mod, '').split('_')
    return([mod, cv, rep])


# +
rf_yhats_groups = pd.concat([
    # get mod, cv, rep for all rf_yhats, set col to be the ith file processed
    pd.DataFrame([parse_yhat_filename(e) for e in rf_yhats][i]).rename(columns = {0:i}) 
    # merge dfs
    for i in range(len(rf_yhats))], axis = 1)

rf_yhats_groups = rf_yhats_groups.T.rename(columns = {0:'Mod', 1:'CV', 2:'Rep'})
rf_yhats_groups['File'] = rf_yhats

rf_yhats_groups.head()

# +
xgb_yhats_groups = pd.concat([
    # get mod, cv, rep for all xgb_yhats, set col to be the ith file processed
    pd.DataFrame([parse_yhat_filename(e) for e in xgb_yhats][i]).rename(columns = {0:i}) 
    # merge dfs
    for i in range(len(xgb_yhats))], axis = 1)

xgb_yhats_groups = xgb_yhats_groups.T.rename(columns = {0:'Mod', 1:'CV', 2:'Rep'})
xgb_yhats_groups['File'] = xgb_yhats

xgb_yhats_groups.head()

# +
hist_yhats_groups = pd.concat([
    # get mod, cv, rep for all hist_yhats, set col to be the ith file processed
    pd.DataFrame([parse_yhat_filename(e) for e in hist_yhats][i]).rename(columns = {0:i}) 
    # merge dfs
    for i in range(len(hist_yhats))], axis = 1)

hist_yhats_groups = hist_yhats_groups.T.rename(columns = {0:'Mod', 1:'CV', 2:'Rep'})
hist_yhats_groups['File'] = hist_yhats

hist_yhats_groups.head()


# -

# ### Parse files

# +
# Parse the prediction file, reverse scaling and optionally rename (e.g. with the model/cv/rep)

def get_ml_yhats(file_name = 'rf2017_5rfYHats.csv',
                 rename_Yhat = None,
                file_path = shared_out_path):
    df = pd.read_csv(file_path+'/'+file_name)

    # solve for the scaling /centering values
    vals = df.loc[:, ['Yield_Mg_ha', 'YMat']].dropna()
    val_x = vals[['YMat']]
    val_y = vals[['Yield_Mg_ha']]

    reg = LinearRegression().fit(val_x, val_y)
    center_val = reg.intercept_[0]
    scale_val = reg.coef_[0][0]

    df['Center'] = center_val
    df['Scale'] = scale_val

    df['YHat_Mg_ha'] = (df['YHat']*scale_val) + center_val
    df = df.loc[:, ['Env', 'Year', 'Hybrid', 'Yield_Mg_ha', 'YHat_Mg_ha']]

    if rename_Yhat == None:
        return(df)
    else:
        return(df.rename(columns = {'YHat_Mg_ha': rename_Yhat}))



# +
# aggregate all rf predictions
# pull select cols from each file then merge
rf_yhats_df = [get_ml_yhats(file_name = rf_yhat,
                                      rename_Yhat = rf_yhat.replace('rfYHats.csv', '')
                ) for rf_yhat in tqdm.tqdm(rf_yhats)]

# This is messy but the alternative is to repeatedly merge
# drop select cols from all but 0th data frame so they are not duplicated in 
# in the yhat df
rf_yhats_df = pd.concat([rf_yhats_df[e] if e == 0 else 
                         rf_yhats_df[e].drop(columns = [
                             'Env', 'Year', 'Hybrid', 'Yield_Mg_ha']) 
                         for e in range(0, len(rf_yhats_df))], axis = 1)

# +
# Repeat for xgb
xgb_yhats_df = [get_ml_yhats(file_name = xgb_yhat,
                                      rename_Yhat = xgb_yhat.replace('xgbYHats.csv', '')
                ) for xgb_yhat in tqdm.tqdm(xgb_yhats)]

xgb_yhats_df = pd.concat([xgb_yhats_df[e] if e == 0 else 
                         xgb_yhats_df[e].drop(columns = [
                             'Env', 'Year', 'Hybrid', 'Yield_Mg_ha']) 
                         for e in range(0, len(xgb_yhats_df))], axis = 1)
# -

xgb_yhats_df

# +
# Repeat for hist
hist_yhats_df = [get_ml_yhats(file_name = hist_yhat,
                                      rename_Yhat = hist_yhat.replace('histYHats.csv', '')
                ) for hist_yhat in tqdm.tqdm(hist_yhats)]

hist_yhats_df = pd.concat([hist_yhats_df[e] if e == 0 else 
                         hist_yhats_df[e].drop(columns = [
                             'Env', 'Year', 'Hybrid', 'Yield_Mg_ha']) 
                         for e in range(0, len(hist_yhats_df))], axis = 1)
# -

hist_yhats_df = hist_yhats_df.drop_duplicates()

# ### Agg extracted predictions

# +
yHats_df = pd.concat([
    phno, 
    rf_yhats_df.drop(columns =  ['Env', 'Year', 'Hybrid', 'Yield_Mg_ha']),
    xgb_yhats_df.drop(columns = ['Env', 'Year', 'Hybrid', 'Yield_Mg_ha'])
], axis = 1)

yHats_df.head()
# -

# note constraining to entries with historical data reduces the number of obs #--NOTE--
# not done for submisson 1
yHats_df = yHats_df.merge(hist_yhats_df)

yHats_notBlup = yHats_df


# ## Blup Version

# +
def reformat_blup_filename(file_name = 'Ws20141BlupYHats.csv'):
    file_name = file_name.replace('BlupYHats.csv', '')
    file_digits = re.findall('\d+', file_name)[0]  
    file_class  = re.findall('\D+', file_name)[0]   
    file_year = file_digits[0:4]
    file_rep = file_digits[-1]
    return(file_class+file_year+'_'+file_rep)
        
# reformat_blup_filename(file_name = 'Ws20141BlupYHats.csv')


# +
# get predictions in the same format as for the ml models. 
# Note that it contains the functionality to return or not return the index in
# phno for confirming data is appropriately matched.
def get_blup_yhats(file_name = 'Ws20141BlupYHats.csv',
                   rename_Yhat = None, #'Ws2014_1'
                   return_Ind = True
                  ):
    df = pd.read_csv(shared_out_path + '/'+ file_name
                    ).rename(columns = {'Unnamed: 0': 'Ind',
                                       'Yield_Mg_ha': 'Yield_Mg_ha_Scaled',
                                       'YHat': 'YHat_Scaled'})
    blup_yhats_dig = re.findall('\d+', file_name)[0]    
    df['Class'] = re.findall('\D+', file_name)[0]   
    df['CV'] = blup_yhats_dig[0:4]
    df['Rep'] = blup_yhats_dig[-1]
    ref = pd.read_csv('./data/Processed/phno_ref_'+str(blup_yhats_dig[0:4])+'_small.csv'
               ).rename(columns = {'Unnamed: 0': 'Ind'})

    df = ref.merge(df, how = 'outer')

    # solve for the scaling /centering values
    vals = df.loc[:, ['Yield_Mg_ha', 'Yield_Mg_ha_Scaled']].dropna()
    val_x = vals[['Yield_Mg_ha_Scaled']]
    val_y = vals[['Yield_Mg_ha']]

    reg = LinearRegression().fit(val_x, val_y)
    center_val = reg.intercept_[0]
    scale_val = reg.coef_[0][0]

    df['Center'] = center_val
    df['Scale'] = scale_val

    df['YHat_Mg_ha'] = (df['YHat_Scaled']*scale_val) + center_val
    
    if return_Ind:
        df = df.loc[:, ['Ind', 'Env', 'Hybrid', 'Year', 'Yield_Mg_ha', 'YHat_Mg_ha']]
    else:
        df = df.loc[:, [       'Env', 'Hybrid', 'Year', 'Yield_Mg_ha', 'YHat_Mg_ha']]

    if rename_Yhat == None:
        return(df)
    else:
        return(df.rename(columns = {'YHat_Mg_ha': rename_Yhat}))

# get_blup_yhats(file_name = 'Ws20141BlupYHats.csv',
#                rename_Yhat = 'Ws2014_1')


# +
# Repeat pattern for differently formatted data
blup_yhats_df = [get_blup_yhats(
    file_name = e,
    rename_Yhat = reformat_blup_filename(file_name = e),
#     return_Ind = False
    return_Ind = True
) for e in blup_yhats]

blup_yhats_df = pd.concat([blup_yhats_df[e] if e == 0 else 
                         blup_yhats_df[e].drop(columns = [
                             e for e in list(blup_yhats_df[e]) if e in [
                                 'Ind', 'Env', 'Year', 'Hybrid', 'Yield_Mg_ha'
                             ]]) 
                         for e in range(0, len(blup_yhats_df))], axis = 1)

# +
# Make sure there's at most one prediction for each set of identifiers
# replace with a filler number so that when it I group on yield these rows aren't dropped
mask = (blup_yhats_df.Year == 2022)
blup_yhats_df.loc[mask, 'Yield_Mg_ha'] = -9999

blup_yhats_df = blup_yhats_df.drop(columns = ['Ind']
                            ).groupby(['Env', 'Hybrid', 'Year', 'Yield_Mg_ha']
                            ).agg(np.mean).reset_index()

# undo
mask = (blup_yhats_df.Year == 2022)
blup_yhats_df.loc[mask, 'Yield_Mg_ha'] = np.nan
# confirm there are no negative yield values before proceeding
assert np.nanmin(blup_yhats_df.Yield_Mg_ha) > 0 
# -

# ### Agg extracted predictions

# +
# note there are more entries per keys in phno than in blup_yhats_df. That's okay.
tmp = blup_yhats_df.merge(phno.reset_index()).rename(columns = {'index':'phno_Idx'})
tmp = tmp.merge(rf_yhats_df, how = 'left').drop_duplicates()
tmp = tmp.merge(xgb_yhats_df, how = 'left').drop_duplicates()

tmp = tmp.merge(hist_yhats_df, how = 'left').drop_duplicates()
# TODO add more -- 

# reorder cols
first_cols = ['phno_Idx', 'Env', 'Hybrid', 'Year', 'Yield_Mg_ha']
tmp = tmp.loc[:, first_cols+[e for e in list(tmp) if e not in first_cols]]
tmp.head()
# 22791  
# -

yHats_yesBlup = tmp

# # Prepare covariates

# +
save_path = './data/Processed/'

# phno = pd.read_csv(save_path+"phno3.csv")

YMat = np.load(save_path+'YMat3.npy')
GMat = np.load(save_path+'GMat3.npy')
SMat = np.load(save_path+'SMat3.npy')
WMat = np.load(save_path+'WMat3.npy')
MMat = np.load(save_path+'MMat3.npy')

GMatNames = np.load(save_path+'GMatNames.npy')
SMatNames = np.load(save_path+'SMatNames.npy')
WMatNames = np.load(save_path+'WMatNames.npy')
MMatNames = np.load(save_path+'MMatNames.npy')

# +
# create backups in case I overwrite the covariate matrices
phno_backup = phno

YMat_backup = YMat
GMat_backup = GMat
SMat_backup = SMat
WMat_backup = WMat
MMat_backup = MMat

GMatNames_backup = GMatNames
SMatNames_backup = SMatNames
WMatNames_backup = WMatNames
MMatNames_backup = MMatNames


# +
def restrict_mats(phno_idxs = [], # list of indices to be used. If [] passed make no change
                  phno = phno_backup,
                  YMat = YMat_backup,
                  GMat = GMat_backup,
                  SMat = SMat_backup,
                  WMat = WMat_backup,
                  MMat = MMat_backup):
    # reduce and reset indices
    if phno_idxs == []:
        pass
    else:
        phno = phno.loc[phno_idxs, ].reset_index().rename(columns = {'index': 'phno_idxs'})
        YMat = YMat[phno_idxs]
        GMat = GMat[phno_idxs]
        SMat = SMat[phno_idxs]
        WMat = WMat[phno_idxs]
        MMat = MMat[phno_idxs]
    return(phno, YMat, GMat, SMat, WMat, MMat)

# Restrict to obs used for BLUPs
# phno, YMat, GMat, SMat, WMat, MMat = restrict_mats(
#     phno_idxs = list(tmp.phno_Idx), # list of indices to be used. If [] passed make no change
#     phno = phno_backup,
#     YMat = YMat_backup,
#     GMat = GMat_backup,
#     SMat = SMat_backup,
#     WMat = WMat_backup,
#     MMat = MMat_backup)


# -

# # Ensembling Models

# ## yHats_yesBlup

yHats_yesBlup

phno, YMat, GMat, SMat, WMat, MMat = restrict_mats(
    phno_idxs = list(yHats_yesBlup.phno_Idx), # list of indices to be used. If [] passed make no change
    phno = phno_backup,
    YMat = YMat_backup,
    GMat = GMat_backup,
    SMat = SMat_backup,
    WMat = WMat_backup,
    MMat = MMat_backup)

# ### Simplest thing that might work
#
# Average with respect to 
# Within model type
# within hold out year
# across hold out years

# +
# ens_cols = [e for e in list(yHats_yesBlup) if e not in ['phno_Idx', 'Env', 'Hybrid', 'Year', 'Yield_Mg_ha']]
# This is hardcoded to ensure the results are reproducible
ens_cols = ['As2021_1', 'Ws2014_1', 'As2018_3', 'As2019_2', 'Ws2021_1', 'As2017_2', 
            'Ws2017_1', 'Ws2020_1', 'As2014_1', 'Ws2018_1', 'As2016_3', 'Ws2017_3', 
            'As2018_1', 'Ws2019_3', 'As2018_2', 'Ws2021_3', 'Ws2014_3', 'Ws2019_1', 
            'Ws2015_3', 'As2014_2', 'Ws2021_2', 'Ws2018_2', 'As2015_3', 'As2020_1', 
            'As2019_1', 'Ws2016_1', 'As2020_3', 'Ws2018_3', 'Ws2014_2', 'As2016_1', 
            'Ws2015_2', 'As2021_3', 'Ws2015_1', 'Ws2017_2', 'As2019_3', 'As2016_2', 
            'As2014_3', 'Ws2020_3', 'As2015_1', 'Ws2016_3', 'As2017_3', 'As2017_1', 
            'Ws2019_2', 'As2020_2', 'Ws2020_2', 'Ws2016_2', 'As2015_2', 'As2021_2', 
            'rf2017_5', 'rf2016_8', 'rf2015_4', 'rf2019_3', 'rf2020_8', 'rf2016_1', 
            'rf2019_9', 'rf2015_1', 'rf2020_3', 'rf2015_7', 'rf2021_6', 'rf2018_8', 
            'rf2017_7', 'rf2021_9', 'rf2015_5', 'rf2020_1', 'rf2017_8', 'rf2015_2', 
            'rf2015_0', 'rf2014_8', 'rf2020_5', 'rf2019_1', 'rf2020_4', 'rf2016_0', 
            'rf2014_9', 'rf2021_4', 'rf2014_4', 'rf2021_2', 'rf2016_3', 'rf2014_5', 
            'rf2019_4', 'rf2020_6', 'rf2019_2', 'rf2017_4', 'rf2016_6', 'rf2018_0', 
            'rf2020_0', 'rf2015_3', 'rf2021_0', 'rf2019_7', 'rf2018_1', 'rf2016_7', 
            'rf2021_5', 'rf2019_6', 'rf2014_3', 'rf2017_2', 'rf2018_4', 'rf2018_7', 
            'rf2017_9', 'rf2015_8', 'rf2018_6', 'rf2017_0', 'rf2021_1', 'rf2014_6', 
            'rf2020_7', 'rf2016_9', 'rf2018_3', 'rf2016_5', 'rf2017_1', 'rf2017_6', 
            'rf2021_7', 'rf2018_9', 'rf2014_7', 'rf2015_6', 'rf2019_8', 'rf2014_0', 
            'rf2016_2', 'rf2021_8', 'rf2020_9', 'rf2018_5', 'rf2015_9', 'rf2018_2', 
            'rf2014_2', 'rf2021_3', 'rf2016_4', 'rf2020_2', 'rf2014_1', 'rf2017_3', 
            'rf2019_5', 'rf2019_0', 'xgb2015_1', 'xgb2021_4', 'xgb2021_7', 'xgb2015_5', 
            'xgb2016_9', 'xgb2020_4', 'xgb2017_3', 'xgb2018_1', 'xgb2017_5', 'xgb2019_3', 
            'xgb2018_0', 'xgb2021_9', 'xgb2014_1', 'xgb2020_0', 'xgb2020_9', 'xgb2018_7', 
            'xgb2018_8', 'xgb2015_4', 'xgb2016_0', 'xgb2021_1', 'xgb2018_4', 'xgb2019_8', 
            'xgb2015_7', 'xgb2020_2', 'xgb2017_1', 'xgb2016_8', 'xgb2014_3', 'xgb2021_2', 
            'xgb2016_1', 'xgb2019_5', 'xgb2017_8', 'xgb2018_3', 'xgb2021_6', 'xgb2015_0', 
            'xgb2019_1', 'xgb2018_5', 'xgb2014_5', 'xgb2019_9', 'xgb2020_6', 'xgb2020_8', 
            'xgb2015_3', 'xgb2019_2', 'xgb2014_6', 'xgb2016_2', 'xgb2017_7', 'xgb2016_4', 
            'xgb2021_0', 'xgb2014_2', 'xgb2014_4', 'xgb2016_3', 'xgb2020_1', 'xgb2019_0', 
            'xgb2021_8', 'xgb2017_2', 'xgb2014_0', 'xgb2019_7', 'xgb2019_4', 'xgb2018_6', 
            'xgb2018_9', 'xgb2014_8', 'xgb2020_3', 'xgb2016_6', 'xgb2017_6', 'xgb2016_7', 
            'xgb2021_5', 'xgb2017_9', 'xgb2019_6', 'xgb2017_4', 'xgb2016_5', 'xgb2020_7', 
            'xgb2015_9', 'xgb2018_2', 'xgb2020_5', 'xgb2014_7', 'xgb2017_0', 'xgb2015_8', 
            'xgb2021_3', 'xgb2014_9', 'xgb2015_6', 'xgb2015_2']


ens_cols_info = pd.DataFrame(zip(
    ens_cols,
    [re.findall('\d\d\d\d', e)[0] for e in list(ens_cols)],
    [re.findall('^\D+', e)[0] for e in list(ens_cols)],
    [re.findall('\d+$', e)[0] for e in list(ens_cols)]
)).rename(columns = {0:'Model_Name', 1:'Holdout_Year', 2:'Model', 3:'Rep',})

ens_cols_info

tmp = yHats_yesBlup

id_vars = ['phno_Idx', 'Env', 'Hybrid', 'Year']

tmp = tmp.melt(id_vars=id_vars, 
               value_vars=[e for e in list(tmp) if e not in id_vars],
               var_name='Model_Name', value_name='yHat')

tmp = tmp.merge(ens_cols_info, how = 'left')

tmp.head(5)
# -

# average replicates
tmp = tmp.groupby(id_vars+['Holdout_Year', 'Model']).agg(yHat = ('yHat', np.mean)).reset_index().drop_duplicates()
tmp.head()

# average over model types
tmp = tmp.groupby(id_vars+['Holdout_Year']).agg(yHat = ('yHat', np.mean)).reset_index().drop_duplicates()
tmp.head()

# average over holdout years
tmp = tmp.groupby(id_vars).agg(yHat = ('yHat', np.mean)).reset_index().drop_duplicates()
tmp.head()

tmp


def format_submission(df = tmp,
                      yhat_col = 'yHat'
):
    sub_template = pd.read_csv('./data/Maize_GxE_Competition_Data/Testing_Data/1_Submission_Template_2022.csv')
    tmp = sub_template.drop(columns = ['Yield_Mg_ha']).merge(df, how = 'left')
    tmp['Yield_Mg_ha'] = tmp[yhat_col]
    tmp = tmp.loc[:, ['Env', 'Hybrid', 'Yield_Mg_ha']]
    return(tmp)


first_submission = format_submission(
    df = tmp,
    yhat_col = 'yHat')

if False:
    first_submission.to_csv('./notebook_artifacts/99_submissions/Submission_1.csv', index = 'False')

# ### Test with RF (unoptimized)

yHats_yesBlup

EMat = np.array(yHats_yesBlup.drop(columns = ['phno_Idx', 'Env', 'Hybrid', 'Year', 'Yield_Mg_ha']))
EMatNames =list(yHats_yesBlup.drop(columns = ['phno_Idx', 'Env', 'Hybrid', 'Year', 'Yield_Mg_ha']))


# +
# transform to panel data
def wthr_rank_3to2(x_3d):
    n_obs, n_days, n_metrics = x_3d.shape
    return(x_3d.reshape(n_obs, (n_days*n_metrics)))

def wthr_features_rank_2to3(x_3d, feature_import):
    n_obs, n_days, n_metrics = x_3d.shape
    return(feature_import.reshape(n_days, n_metrics))

def y_rank_2to1(y_2d):
    n_obs = y_2d.shape[0]
    return(y_2d.reshape(n_obs, ))


# -

def prep_ensemble_train_test(
    test_this_year = '2014',
    downsample = False, 
    downsample_train = 1000,
    downsample_test  =  100,
    phno = phno,
    GMat = GMat,
    SMat = SMat,
    WMat = WMat,
    MMat = MMat,
    EMat = EMat, # <----------------------- Add ensemble matrix
    YMat = YMat
):

    mask_undefined = (phno.Yield_Mg_ha.isna()) # these can be imputed but not evaluated
    mask_test = ((phno.Year == int(test_this_year)) & (~mask_undefined))
    mask_train = ((phno.Year != int(test_this_year)) & (~mask_undefined))
    test_idx = phno.loc[mask_test, ].index
    train_idx = phno.loc[mask_train, ].index

    if downsample:
        train_idx = np.random.choice(train_idx, downsample_train)
        test_idx = np.random.choice(test_idx, downsample_test)


    # Get Scalings ---------------------------------------------------------------
    YMat_center = np.mean(YMat[train_idx], axis = 0)
    YMat_scale = np.std(YMat[train_idx], axis = 0)

    SMat_center = np.mean(SMat[train_idx, :], axis = 0)
    SMat_scale = np.std(SMat[train_idx, :], axis = 0)

    WMat_center = np.mean(WMat[train_idx, :, :], axis = 0)
    WMat_scale = np.std(WMat[train_idx, :, :], axis = 0)

    MMat_center = np.nanmean(MMat[train_idx, :], axis = 0)
    MMat_scale = np.nanstd(MMat[train_idx, :], axis = 0)
    # if std is 0, set to 1
    MMat_scale[MMat_scale == 0] = 1

    EMat_center = np.mean(EMat[train_idx, :], axis = 0)
    EMat_scale = np.std(EMat[train_idx, :], axis = 0)
    
    # Center and Scale -----------------------------------------------------------
    YMat = (YMat - YMat_center)/YMat_scale
    SMat = (SMat - SMat_center)/SMat_scale
    MMat = (MMat - MMat_center)/MMat_scale
    EMat = (EMat - EMat_center)/EMat_scale

    # Split ------------------------------------------------------------------
    train_g = GMat[train_idx, :]
    test_g  = GMat[test_idx, :]

    train_s = SMat[train_idx, :]
    test_s  = SMat[test_idx, :]

    train_w = WMat[train_idx, :, :]
    test_w  = WMat[test_idx, :, :]

    train_m = MMat[train_idx, :]
    test_m  = MMat[test_idx, :]

    train_y = YMat[train_idx]
    test_y  = YMat[test_idx]

    train_e = EMat[train_idx, :]
    test_e  = EMat[test_idx, :]
    
    # Reshape to rank 1
    train_y = train_y.reshape([train_y.shape[0], 1])
    test_y = test_y.reshape([test_y.shape[0], 1])

    # GSWM
    train_x_2d = np.concatenate([train_g, train_s, wthr_rank_3to2(x_3d = train_w), train_m, train_e], axis = 1)
    train_y_1d = y_rank_2to1(y_2d = train_y)
    test_x_2d = np.concatenate([test_g, test_s, wthr_rank_3to2(x_3d = test_w), test_m, test_e], axis = 1)
    test_y_1d = y_rank_2to1(y_2d = test_y)
    
    full_x_2d = np.concatenate([GMat, SMat, wthr_rank_3to2(x_3d = WMat), MMat, EMat], axis = 1)
    return(train_x_2d, train_y_1d, test_x_2d, test_y_1d, full_x_2d, YMat_center, YMat_scale, YMat)


test_x_2d_names = list(GMatNames)+list(SMatNames)+list(WMatNames)+list(MMatNames)+list(EMatNames)


# find name in EMatNames that have the test year, all others should be removed
# e.g. if cv == 2021, then the model should only have yhats that were produced
#      with a test year of 2021
def get_non_cv_cols(test_this_year = 2022,
                    EMatNames = EMatNames,
                    test_x_2d_names = test_x_2d_names):
    rm_cols = [e for e in EMatNames if not re.match('\D+'+str(test_this_year)+'_\d', e)]
    select_col_idxs = [i for i in range(len(test_x_2d_names)) if test_x_2d_names[i] not in rm_cols]
    return(select_col_idxs)


# +
# Setup ----------------------------------------------------------------------
trial_name = 'enybr' # ensemble, yes blup, random forest
n_trials= 120 #FIXME
n_jobs = 30  #FIXME

downsample = False
downsample_train = 1000
downsample_test  =  100

def objective(trial): 
    rf_max_depth = trial.suggest_int('rf_max_depth', 2, 100, log=True)
    rf_n_estimators = trial.suggest_int('rf_n_estimators', 20, 100, log=True)
    rf_min_samples_split = trial.suggest_float('rf_min_samples_split', 0.005, 0.5, log=True)
    
    regr = RandomForestRegressor(
        max_depth = rf_max_depth, 
        n_estimators = rf_n_estimators,
        min_samples_split = rf_min_samples_split
        )
    
#     rf = regr.fit(train_x_2d, train_y_1d)
#     return (mean_squared_error(train_y_1d, rf.predict(train_x_2d), squared=False))
    rf = regr.fit(test_x_2d, test_y_1d)
    return (mean_squared_error(test_y_1d, rf.predict(test_x_2d), squared=False))


if False:
    reset_trial_name = trial_name
    print("""
    ------------------------------------------------------------------------------
               Note: Ensemble fit using previous test fold.
    ------------------------------------------------------------------------------
    """)
    for test_this_year in ['2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021']:
        print("""
    ------------------------------------------
    ------------------"""+test_this_year+"""------------------
        """)    

        trial_name = reset_trial_name
        trial_name = trial_name+test_this_year
        print(test_this_year)
        # Data Prep. -----------------------------------------------------------------
        # Set up train/test indices --------------------------------------------------
        train_x_2d, train_y_1d, test_x_2d, test_y_1d, full_x_2d, YMat_center, YMat_scale, YMat = prep_ensemble_train_test(
                test_this_year = test_this_year,
                downsample = downsample, 
                downsample_train = downsample_train,
                downsample_test  =  downsample_test,
                phno = phno,
                GMat = GMat,
                SMat = SMat,
                WMat = WMat,
                MMat = MMat,
                EMat = EMat,
                YMat = YMat)

        test_year_cols = get_non_cv_cols(
            test_this_year = test_this_year,
            EMatNames = EMatNames,
            test_x_2d_names = test_x_2d_names)

        # excise columns that would allow for information leakage into the 
        train_x_2d = train_x_2d[:, np.array(test_year_cols)]
        test_x_2d  =  test_x_2d[:, np.array(test_year_cols)]
        full_x_2d  =  full_x_2d[:, np.array(test_year_cols)]


        # HPS Study ------------------------------------------------------------------
        cache_save_name = cache_path+trial_name+'_hps.pkl'
        if os.path.exists(cache_save_name):
            study = pkl.load(open(cache_save_name, 'rb'))  
        else:
            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials= n_trials, n_jobs = n_jobs)
            # save    
            pkl.dump(study, open(cache_save_name, 'wb'))    

        print("""
    --------------Study Complete--------------
    ------------------------------------------
        """)
        def fit_single_rep(Rep = 1):
            # Fit Best HPS --------------------------------------------------------------- 
            cache_save_name = cache_path+trial_name+'_'+str(Rep)+'_mod.pkl'

            # Load cached model if it exists
            if os.path.exists(cache_save_name):
                rf = pkl.load(open(cache_save_name, 'rb'))  
            else:
                regr = RandomForestRegressor(
                        max_depth = study.best_trial.params['rf_max_depth'], 
                        n_estimators = study.best_trial.params['rf_n_estimators'],
                        min_samples_split = study.best_trial.params['rf_min_samples_split']
                        )
    #             rf = regr.fit(train_x_2d, train_y_1d)
                rf = regr.fit(test_x_2d, test_y_1d)
                # save    
                pkl.dump(rf, open(cache_save_name, 'wb'))   

            # Record Predictions -----------------------------------------------------
            out = phno.copy()
            out['YHat'] = rf.predict(full_x_2d)
            out['YMat'] = YMat
            out['Y_center'] = YMat_center
            out['Y_scale'] = YMat_scale
            out['Class'] = trial_name
            out['CV'] = test_this_year
            out['Rep'] = Rep
            out.to_csv('./notebook_artifacts/5_ensembling/'+trial_name+'_'+str(Rep)+'YHats.csv')
    #         out.to_csv('./data/Shared_Model_Output/'+trial_name+'_'+str(Rep)+'rfYHats.csv')

        # use joblib to get replicate models all at once
        Parallel(n_jobs=10)(delayed(fit_single_rep)(Rep = i) for i in range(10))
# -

enybr_yhats = [e for e in os.listdir(cache_path) if re.match('enybr\d\d\d\d_\dYHats.csv', e)] 

# +
# aggregate all enybr predictions
# pull select cols from each file then merge
enybr_yhats_df = [get_ml_yhats(file_name = enybr_yhat,
                               rename_Yhat = enybr_yhat.replace('YHats.csv', ''),
                               file_path = cache_path
                ) for enybr_yhat in tqdm.tqdm(enybr_yhats)]

# This is messy but the alternative is to repeatedly merge
# drop select cols from all but 0th data frame so they are not duplicated in 
# in the yhat df
enybr_yhats_df = pd.concat([enybr_yhats_df[e] if e == 0 else 
                         enybr_yhats_df[e].drop(columns = [
                             'Env', 'Year', 'Hybrid', 'Yield_Mg_ha']) 
                         for e in range(0, len(enybr_yhats_df))], axis = 1)
# -

enybr_yhats_df_backup = enybr_yhats_df.copy()
enybr_yhats_df_backup


# +
# returns uniform weights across years and replicates within years

def ens_unif_yearwise_weight(current_year, 
                            df):
    mask = (df.Year == current_year)
#     y_true = df.loc[mask, 'Yield_Mg_ha']
    current_cols = [e for e in list(df) if re.match('\D+'+str(current_year)+'.+', e)]

    out = pd.DataFrame(zip(
        [current_year for i in current_cols],
#         [current_year for i in range(len(current_cols))],
        current_cols,
        [1/len(current_cols) for current_col in current_cols]
    )
    ).rename(columns = {0:'Year',
                        1:'Ensemble',
                        2:'fracYear'}
            )
    return(out)


ens_weight_df = pd.concat([
    ens_unif_yearwise_weight(
    current_year = e,
    df = enybr_yhats_df) for e in [2021, 2020, 2019, 2018, 2017, 2016, 2015, 2014] 
])


ens_weight_df['Weights'] = ens_weight_df['fracYear'] / np.sum(ens_weight_df['fracYear'])

# +
ens_weight_df_unif = ens_weight_df

ens_weight_df.reset_index()


# +
def ens_calc_yearwise_rmses(current_year, 
                            df):
    mask = (df.Year == current_year)
    y_true = df.loc[mask, 'Yield_Mg_ha']
    current_cols = [e for e in list(df) if re.match('\D+'+str(current_year)+'.+', e)]

    out = pd.DataFrame(zip(
        [current_year for i in current_cols],
#         [current_year for i in range(len(current_cols))],
        current_cols,
        [mean_squared_error(y_true, df.loc[mask, current_col], squared = False) for current_col in current_cols]
    )
    ).rename(columns = {0:'Year',
                        1:'Ensemble',
                        2:'TestRMSE'}
            )
    return(out)


ens_weight_df = pd.concat([
    ens_calc_yearwise_rmses(
    current_year = e,
    df = enybr_yhats_df) for e in [2021, 2020, 2019, 2018, 2017, 2016, 2015, 2014] 
])
# -

ens_weight_df['Weights'] = ens_weight_df['TestRMSE']
ens_weight_df['Weights'] = (1/ens_weight_df['Weights'])
ens_weight_df['Weights'] = ens_weight_df['Weights'] / np.sum(ens_weight_df['Weights'])

ens_weight_df

enybr_yhats_df

# +
# Apply weights and calculate new estimates

tmp = enybr_yhats_df

for ensemble in ens_weight_df.Ensemble:
    weight = float(ens_weight_df.loc[(ens_weight_df.Ensemble == ensemble), 'Weights'])
    tmp.loc[:, ensemble] = tmp.loc[:, ensemble] * weight
    

tmp.loc[:, 'Yhat_Mg_ha_Weighted'] = np.sum(tmp.loc[:, ens_weight_df.Ensemble], axis = 1)

# px.scatter(tmp, x = 'Yield_Mg_ha', y = 'Yhat_Mg_ha_Weighted')
# -

enybr_yhats_df

tmp

# possibility -- todo how similar would submission 2 be relative to one with maximal obs
second_possible_yb = format_submission(
    df = tmp,
    yhat_col = 'Yhat_Mg_ha_Weighted')

second_possible_yb

# Uses historical data
# uses rf aggregation
# uses inverse rmse weighting on cv aggs
second_submission = second_possible_yb
if False:
    second_submission.to_csv('./notebook_artifacts/99_submissions/Submission_2.csv', index = 'False')

# Compare the two submissions: 
px.scatter(
    pd.DataFrame(zip(
        first_submission.Yield_Mg_ha, 
        second_possible_yb.Yield_Mg_ha
    )).rename(columns = {0:'s1', 1:'s2'}),
    x = 's1', y = 's2')

## Submission 3 ==============================================================
enybr_yhats_df = enybr_yhats_df_backup
enybr_yhats_df

# +
# Apply weights and calculate new estimates

tmp = enybr_yhats_df

for ensemble in ens_weight_df_unif.Ensemble:
    weight = float(ens_weight_df_unif.loc[(ens_weight_df_unif.Ensemble == ensemble), 'Weights'])
    tmp.loc[:, ensemble] = tmp.loc[:, ensemble] * weight
    

tmp.loc[:, 'Yhat_Mg_ha_Weighted'] = np.sum(tmp.loc[:, ens_weight_df_unif.Ensemble], axis = 1)
# -

third_possible_yb = format_submission(
    df = tmp,
    yhat_col = 'Yhat_Mg_ha_Weighted')

# Uses historical data
# uses rf aggregation
# uses uniform weighting on cv aggs
third_submission = third_possible_yb
if False:
    third_submission.to_csv('./notebook_artifacts/99_submissions/Submission_3.csv', index = 'False')
    third_submission.to_csv('./notebook_artifacts/99_submissions/Submission_3_enybr_uw.csv')

# Compare the two submissions: 
px.scatter(
    pd.DataFrame(zip(
        third_possible_yb.Yield_Mg_ha, 
        second_possible_yb.Yield_Mg_ha
    )).rename(columns = {0:'s1', 1:'s2'}),
    x = 's1', y = 's2')

# ### Feed into DNN

EMatNames[-1]

[e.shape for e in [yHats_yesBlup, yHats_notBlup]]

[e.shape for e in [phno, YMat, GMat, SMat, WMat, MMat, EMat]]
# yHats_yesBlup, 

# ## yHats_noBlup

yHats_notBlup

idxs_with_hist = list(
    yHats_notBlup.merge(phno.reset_index(), 
                        how = 'inner'
                       ).drop_duplicates().loc[:, 'index'])

phno, YMat, GMat, SMat, WMat, MMat = restrict_mats(
    phno_idxs = idxs_with_hist, #[], # list of indices to be used. If [] passed make no change
    # here restrict to suset that historical coud be calc-ed for
    phno = phno_backup,
    YMat = YMat_backup,
    GMat = GMat_backup,
    SMat = SMat_backup,
    WMat = WMat_backup,
    MMat = MMat_backup)

# ### Simplest thing that might work
#
# Average with respect to 
# Within model type
# within hold out year
# across hold out years

# +
ens_cols = [e for e in list(yHats_yesBlup) if e not in ['phno_Idx', 'Env', 'Hybrid', 'Year', 'Yield_Mg_ha']]
# Specifying prefix allows for reproducibility without hardcoding
ens_cols = [e for e in ens_cols if  re.match('^[rf\d\d\d\d_\d|xgb\d\d\d\d_\d]', e)]
ens_cols

ens_cols_info = pd.DataFrame(zip(
    ens_cols,
    [re.findall('\d\d\d\d', e)[0] for e in list(ens_cols)],
    [re.findall('^\D+', e)[0] for e in list(ens_cols)],
    [re.findall('\d+$', e)[0] for e in list(ens_cols)]
)).rename(columns = {0:'Model_Name', 1:'Holdout_Year', 2:'Model', 3:'Rep',})

ens_cols_info

tmp = yHats_yesBlup

id_vars = ['phno_Idx', 'Env', 'Hybrid', 'Year']

tmp = tmp.melt(id_vars=id_vars, 
               value_vars=[e for e in list(tmp) if e not in id_vars],
               var_name='Model_Name', value_name='yHat')

tmp = tmp.merge(ens_cols_info, how = 'left')

tmp.head(5)
# -

# average replicates
tmp = tmp.groupby(id_vars+['Holdout_Year', 'Model']).agg(yHat = ('yHat', np.mean)).reset_index().drop_duplicates()
tmp.head()

# average over model types
tmp = tmp.groupby(id_vars+['Holdout_Year']).agg(yHat = ('yHat', np.mean)).reset_index().drop_duplicates()
tmp.head()

# average over holdout years
tmp = tmp.groupby(id_vars).agg(yHat = ('yHat', np.mean)).reset_index().drop_duplicates()
tmp.head()

tmp

first_counterfactual_submission = format_submission(
    df = tmp,
    yhat_col = 'yHat')

# Compare the two submissions: 
px.scatter(
    pd.DataFrame(zip(
        first_submission.Yield_Mg_ha, 
        first_counterfactual_submission.Yield_Mg_ha
    )).rename(columns = {0:'s1', 1:'s1prime'}),
    x = 's1', y = 's1prime')

# all non blups
# uniform weights (better than the weights in sub 2)
if False:
    # first_counterfactual_submission.to_csv('./notebook_artifacts/99_submissions/Submission_4.csv', index = 'False')
    first_counterfactual_submission.to_csv('./notebook_artifacts/99_submissions/Submission_4_ennbu_uw.csv')

# ### Test with RF (unoptimized)

yHats_notBlup

EMat = np.array(yHats_notBlup.drop(columns = [#'phno_Idx', 
    'Env', 'Hybrid', 'Year', 'Yield_Mg_ha']))
EMatNames =list(yHats_notBlup.drop(columns = [#'phno_Idx', 
    'Env', 'Hybrid', 'Year', 'Yield_Mg_ha']))

test_x_2d_names = list(GMatNames)+list(SMatNames)+list(WMatNames)+list(MMatNames)+list(EMatNames)

# +
# Setup ----------------------------------------------------------------------
trial_name = 'ennbr' # ensemble, no blup, random forest
n_trials= 40 #FIXME
n_jobs = 20  #FIXME

downsample = False
downsample_train = 1000
downsample_test  =  100

def objective(trial): 
    rf_max_depth = trial.suggest_int('rf_max_depth', 2, 100, log=True)
    rf_n_estimators = trial.suggest_int('rf_n_estimators', 20, 100, log=True)
    rf_min_samples_split = trial.suggest_float('rf_min_samples_split', 0.005, 0.5, log=True)
    
    regr = RandomForestRegressor(
        max_depth = rf_max_depth, 
        n_estimators = rf_n_estimators,
        min_samples_split = rf_min_samples_split
        )
    
#     rf = regr.fit(train_x_2d, train_y_1d)
#     return (mean_squared_error(train_y_1d, rf.predict(train_x_2d), squared=False))
    rf = regr.fit(test_x_2d, test_y_1d)
    return (mean_squared_error(test_y_1d, rf.predict(test_x_2d), squared=False))

if False:
    reset_trial_name = trial_name
    print("""
    ------------------------------------------------------------------------------
               Note: Ensemble fit using previous test fold.
    ------------------------------------------------------------------------------
    """)
    for test_this_year in ['2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021']:
        print("""
    ------------------------------------------
    ------------------"""+test_this_year+"""------------------
        """)    

        trial_name = reset_trial_name
        trial_name = trial_name+test_this_year
        print(test_this_year)
        # Data Prep. -----------------------------------------------------------------
        # Set up train/test indices --------------------------------------------------
        train_x_2d, train_y_1d, test_x_2d, test_y_1d, full_x_2d, YMat_center, YMat_scale, YMat = prep_ensemble_train_test(
                test_this_year = test_this_year,
                downsample = downsample, 
                downsample_train = downsample_train,
                downsample_test  =  downsample_test,
                phno = phno,
                GMat = GMat,
                SMat = SMat,
                WMat = WMat,
                MMat = MMat,
                EMat = EMat,
                YMat = YMat)

        test_year_cols = get_non_cv_cols(
            test_this_year = test_this_year,
            EMatNames = EMatNames,
            test_x_2d_names = test_x_2d_names)

        # excise columns that would allow for information leakage into the 
        train_x_2d = train_x_2d[:, np.array(test_year_cols)]
        test_x_2d  =  test_x_2d[:, np.array(test_year_cols)]
        full_x_2d  =  full_x_2d[:, np.array(test_year_cols)]


        # HPS Study ------------------------------------------------------------------
        cache_save_name = cache_path+trial_name+'_hps.pkl'
        if os.path.exists(cache_save_name):
            study = pkl.load(open(cache_save_name, 'rb'))  
        else:
            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials= n_trials, n_jobs = n_jobs)
            # save    
            pkl.dump(study, open(cache_save_name, 'wb'))    

        print("""
    --------------Study Complete--------------
    ------------------------------------------
        """)
        def fit_single_rep(Rep = 1):
            # Fit Best HPS --------------------------------------------------------------- 
            cache_save_name = cache_path+trial_name+'_'+str(Rep)+'_mod.pkl'

            # Load cached model if it exists
            if os.path.exists(cache_save_name):
                rf = pkl.load(open(cache_save_name, 'rb'))  
            else:
                regr = RandomForestRegressor(
                        max_depth = study.best_trial.params['rf_max_depth'], 
                        n_estimators = study.best_trial.params['rf_n_estimators'],
                        min_samples_split = study.best_trial.params['rf_min_samples_split']
                        )
    #             rf = regr.fit(train_x_2d, train_y_1d)
                rf = regr.fit(test_x_2d, test_y_1d)
                # save    
                pkl.dump(rf, open(cache_save_name, 'wb'))   

            # Record Predictions -----------------------------------------------------
            out = phno.copy()
            out['YHat'] = rf.predict(full_x_2d)
            out['YMat'] = YMat
            out['Y_center'] = YMat_center
            out['Y_scale'] = YMat_scale
            out['Class'] = trial_name
            out['CV'] = test_this_year
            out['Rep'] = Rep
            out.to_csv('./notebook_artifacts/5_ensembling/'+trial_name+'_'+str(Rep)+'YHats.csv')
    #         out.to_csv('./data/Shared_Model_Output/'+trial_name+'_'+str(Rep)+'rfYHats.csv')

        # use joblib to get replicate models all at once
        Parallel(n_jobs=10)(delayed(fit_single_rep)(Rep = i) for i in range(10))

# +
ennbr_yhats = [e for e in os.listdir(cache_path) if re.match('ennbr\d\d\d\d_\dYHats.csv', e)] 

# aggregate all ennbr predictions
# pull select cols from each file then merge
ennbr_yhats_df = [get_ml_yhats(file_name = ennbr_yhat,
                               rename_Yhat = ennbr_yhat.replace('YHats.csv', ''),
                               file_path = cache_path
                ) for ennbr_yhat in tqdm.tqdm(ennbr_yhats)]

# This is messy but the alternative is to repeatedly merge
# drop select cols from all but 0th data frame so they are not duplicated in 
# in the yhat df
ennbr_yhats_df = pd.concat([ennbr_yhats_df[e] if e == 0 else 
                         ennbr_yhats_df[e].drop(columns = [
                             'Env', 'Year', 'Hybrid', 'Yield_Mg_ha']) 
                         for e in range(0, len(ennbr_yhats_df))], axis = 1)

ennbr_yhats_df

# +
# uniform weights

ens_weight_df = pd.concat([
    ens_unif_yearwise_weight(
    current_year = e,
    df = ennbr_yhats_df) for e in [2021, 2020, 2019, 2018, 2017, 2016, 2015, 2014] 
])


ens_weight_df['Weights'] = ens_weight_df['fracYear'] / np.sum(ens_weight_df['fracYear'])
ens_weight_df.reset_index()

# +
# Apply weights and calculate new estimates

tmp = ennbr_yhats_df

for ensemble in ens_weight_df.Ensemble:
    weight = float(ens_weight_df.loc[(ens_weight_df.Ensemble == ensemble), 'Weights'])
    tmp.loc[:, ensemble] = tmp.loc[:, ensemble] * weight
    

tmp.loc[:, 'Yhat_Mg_ha_Weighted'] = np.sum(tmp.loc[:, ens_weight_df.Ensemble], axis = 1)

# px.scatter(tmp, x = 'Yield_Mg_ha', y = 'Yhat_Mg_ha_Weighted')
# -

tmp

# possibility -- todo how similar would submission 2 be relative to one with maximal obs
thrid_possible_nb = format_submission(
    df = tmp,
    yhat_col = 'Yhat_Mg_ha_Weighted')

# Compare the two submissions: 
px.scatter(
    pd.DataFrame(zip(
        first_submission.Yield_Mg_ha, 
        thrid_possible_nb.Yield_Mg_ha
    )).rename(columns = {0:'s1', 1:'s1prime'}),
    x = 's1', y = 's1prime')

# No blups
# rf, equal weighting
if False:
    thrid_possible_nb.to_csv('./notebook_artifacts/99_submissions/Submission_4.csv')
    thrid_possible_nb.to_csv('./notebook_artifacts/99_submissions/Submission_4_ennbr_uw.csv')

# Compare the two submissions: 
px.scatter(
    pd.DataFrame(zip(
        first_submission.Yield_Mg_ha, 
        thrid_possible_nb.Yield_Mg_ha
    )).rename(columns = {0:'s1', 1:'s2'}),
    x = 's1', y = 's2')
