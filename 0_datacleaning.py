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

# # Process Data G2F Competition Data. 
#
# > This notebook is the first pass at cleaning the aggregated g2f data. It will draw on code written for past projects, specifically `maizemodel` and `g2fd`. I will also use random forests to help identify which the highest value targets for data cleaning are.

# +
import os, json, requests # for downloading power data with `dl_power_data()`
import glob
import re
import time # add a delay between requests to NASA POWER API for weather data
import pickle as pkl

import numpy as np
from numpy import random

import pandas as pd
pd.set_option('display.max_columns', None)

from sklearn.impute import KNNImputer # for imputing soil for ARH1_2016 & ARH2_2016
from sklearn import preprocessing # LabelEncoder
from sklearn.metrics import mean_squared_error # if squared=False; RMSE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt
import plotly.express as px

import tqdm

import optuna

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# from g2f_comp.internal import *
# -

cache_path = './notebook_artifacts/0_datacleaning/'

# # Data Aggregation

train_dir= "./data/Maize_GxE_Competition_Data/Training_Data/"
test_dir = "./data/Maize_GxE_Competition_Data/Testing_Data/"

# +
# Handy one liners for finding mismatched data types
# find disagreeing types
# [e for e in [e for e in list(x_test) if e in list(x_train)] if (np.dtype(x_train[e]) != np.dtype(x_test[e])) ]
# [(e, np.dtype(x_train[e]), np.dtype(x_test[e])) for e in [e for e in list(x_test) if e in list(x_train)] if (np.dtype(x_train[e]) != np.dtype(x_test[e])) ]

# +
# Aggregate train/test into a single set of dfs
# phenotype/trait
x_train = pd.read_csv(train_dir+'1_Training_Trait_Data_2014_2021.csv')
x_test = pd.read_csv(test_dir+'1_Submission_Template_2022.csv')
# use env to fill in other keys
# x_test[["Field_Location", "Year"]] = x_test['Env'].str.split('_',expand=True)
# x_test["Year"] = x_test["Year"].astype(int)
phno = x_train.merge(x_test, "outer")


# metadata
# I've manually resaved this csv as an xlsx to get around this error with the csv:
# UnicodeDecodeError: 'utf-8' codec can't decode byte 0xca in position 5254: invalid continuation byte
x_train = pd.read_excel(train_dir+'2_Training_Meta_Data_2014_2021.xlsx') 
x_test = pd.read_csv(test_dir+'2_Testing_Meta_Data_2022.csv')
x_test['Date_weather_station_placed'] = pd.to_datetime(x_test.Date_weather_station_placed)
x_test['Date_weather_station_removed'] = pd.to_datetime(x_test.Date_weather_station_removed)
# to string
for e in ['Weather_Station_Serial_Number (Last four digits, e.g. m2700s#####)',
         'Issue/comment_#5', 
          'Issue/comment_#6', 
          'Comments']:
    x_test[e] = x_test[e].astype(str)
meta = x_train.merge(x_test, how = "outer")


# soil
x_train = pd.read_csv(train_dir+'3_Training_Soil_Data_2015_2021.csv')
x_test = pd.read_csv(test_dir+'3_Testing_Soil_Data_2022.csv')

for e in ['E Depth',
 'lbs N/A',
 'Potassium ppm K',
 'Calcium ppm Ca',
 'Magnesium ppm Mg',
 'Sodium ppm Na',
 '%H Sat',
 '%K Sat',
 '%Ca Sat',
 '%Mg Sat',
 '%Na Sat',
 'Mehlich P-III ppm P']:
    x_test[e] = x_test[e].astype(float)

x_test['Comments'] = x_test['Comments'].astype(str)
soil = x_train.merge(x_test, how = "outer")


# weather
x_train = pd.read_csv(train_dir+'4_Training_Weather_Data_2014_2021.csv')
x_test = pd.read_csv(test_dir+'4_Testing_Weather_Data_2022.csv')
wthr = x_train.merge(x_test, how = "outer")

# crop growth model variables (enviromental covariates)
x_train = pd.read_csv(train_dir+'6_Training_EC_Data_2014_2021.csv')
x_test = pd.read_csv(test_dir+'6_Testing_EC_Data_2022.csv')
cgmv = x_train.merge(x_test, how = "outer")


# -

# # Data Cleaning
#
# ## Data Cleaning Functions

# from g2fd
def find_shared_cols(
    df1,# = df,
    df2# = meta
):
    shared_cols = [e for e in list(df1) if e in list(df2)]
    return(shared_cols)


# Changes involving only 1 data frame =======================================
# phno
# object to datetime
phno.Date_Planted = pd.to_datetime(phno.Date_Planted)
phno.Date_Harvested = pd.to_datetime(phno.Date_Harvested)

# Check to make sure there are no missing values for these
assert [] == [e for e in list(cgmv) if 0 != np.mean(cgmv[e].isna())]

# +
# Changes involving _2_ data frames =========================================

# phno and meta share 'Plot_Area_ha', 'Date_Planted' these need to be checked for consistency and moved. 
# use meta to fill in pheno then drop
# meta has the information from 22 and nothing else. Reverse for phno
for e in ['Plot_Area_ha', 'Date_Planted']:
    mask = phno[e].isna()
    fill_ins = phno.loc[mask, ['Env', 'Year']].drop_duplicates().reset_index()

    # for all the unique key combinations find the value that should be inserted and do so.
    for i in range(fill_ins.shape[0]):
        phno_mask = ((phno.Env == fill_ins.loc[i,"Env"]) & 
                    (phno.Year == fill_ins.loc[i,"Year"]) )

        meta_mask = ((meta.Env == fill_ins.loc[i,"Env"]) & 
                    (meta.Year == fill_ins.loc[i,"Year"]) )
        

        
        insert_val = list(meta.loc[meta_mask, e])
        insert_val = [e for e in insert_val if e == e] # get rid of nans 
        
        if insert_val == []:
            pass
        else:
            assert len(insert_val) == 1 # check that there's only one value to imput
            phno.loc[phno_mask, e] = insert_val[0]
# -

meta = meta.drop(columns=['Plot_Area_ha', 'Date_Planted'])


#from g2fd
# generalized version of `sanitize_Experiment_Codes`
def sanitize_col(df, col, simple_renames= {}, split_renames= {}):
    # simple renames
    for e in simple_renames.keys():
        mask = (df[col] == e)
        df.loc[mask, col] = simple_renames[e]

    # splits
    # pull out the relevant multiname rows, copy, rename, append
    for e in split_renames.keys():
        mask = (df[col] == e)
        temp = df.loc[mask, :] 

        df = df.loc[~mask, :]
        for e2 in split_renames[e]:
            temp2 = temp.copy()
            temp2[col] = e2
            df = df.merge(temp2, how = 'outer')

    return(df)


def summarize_col_missing(df):
    return(
        pd.DataFrame({'Col'   : [e for e in list(df)],
              'N_miss' : [sum(df[e].isna()) for e in list(df)],
              'Pr_Comp': [round(100*(1-sum(df[e].isna())/len(df[e])), 1) for e in list(df)]})
    )
# summarize_col_missing(df = meta)


# ## Misc. Column Rearranging and Cleaning
#

# +
rm_Envs = ['GEH1_2020', 'GEH1_2021', 'GEH1_2019' # Germany locations
#           'ARH1_2016', 'ARH2_2016' # Only in 2016, missing soil data
          ]

phno = phno.loc[~phno.Env.isin(rm_Envs), :]
meta = meta.loc[~meta.Env.isin(rm_Envs), :]
soil = soil.loc[~soil.Env.isin(rm_Envs), :]
wthr = wthr.loc[~wthr.Env.isin(rm_Envs), :]
cgmv = cgmv.loc[~cgmv.Env.isin(rm_Envs), :]
# -

# Fix types: object to datetime
phno.Date_Planted = pd.to_datetime(phno.Date_Planted)
phno.Date_Harvested = pd.to_datetime(phno.Date_Harvested)

# +
# drop and redo year and Env
phno = phno.drop(columns = ['Year'])
meta = meta.drop(columns = ['Year'])
soil = soil.drop(columns = ['Year'])


temp = pd.DataFrame(phno['Env']).drop_duplicates().reset_index().drop(columns = 'index')
temp['Env2'] = temp['Env']


temp = sanitize_col(
    df = temp, 
    col = 'Env2', 
    simple_renames= {
    'MOH1_1_2018': 'MOH1-1_2018', 
    'MOH1_2_2018': 'MOH1-2_2018', 
    'MOH1_1_2020': 'MOH1-1_2020', 
    'MOH1_2_2020': 'MOH1-2_2020'
    }, 
    split_renames= {})


assert [] == [e for e in list(temp['Env2']) if len(e.split('_')) != 2]
temp[["Experiment_Code", "Year"]] = temp['Env2'].str.split('_',expand=True)
temp = temp.drop(columns = ['Experiment_Code'])
# -

phno = temp.merge(phno).drop(columns = ['Env']).rename(columns = {'Env2':'Env'})
meta = temp.merge(meta).drop(columns = ['Env']).rename(columns = {'Env2':'Env'})
soil = temp.merge(soil).drop(columns = ['Env']).rename(columns = {'Env2':'Env'})
wthr = temp.merge(wthr).drop(columns = ['Env']).rename(columns = {'Env2':'Env'})
cgmv = temp.merge(cgmv).drop(columns = ['Env']).rename(columns = {'Env2':'Env'})

(find_shared_cols(phno, meta),
 find_shared_cols(phno, soil),
 find_shared_cols(phno, wthr),
 find_shared_cols(phno, cgmv))

# ## Add in missing Enviroments
# confirm there are envs/gps coordinates for everyone

# +
phno_Envs = list(set(phno['Env']))
meta_Envs = list(set(meta['Env']))
soil_Envs = list(set(soil['Env']))
wthr_Envs = list(set(wthr['Env']))
cgmv_Envs = list(set(cgmv['Env']))

all__Envs = list(set(phno_Envs+soil_Envs+wthr_Envs+cgmv_Envs))

def get_missing_envs(data_Envs = []):
    return([e for e in all__Envs if e not in data_Envs])

phno_Envs_miss = get_missing_envs(data_Envs = phno_Envs)
meta_Envs_miss = get_missing_envs(data_Envs = meta_Envs)
soil_Envs_miss = get_missing_envs(data_Envs = soil_Envs)
wthr_Envs_miss = get_missing_envs(data_Envs = wthr_Envs)
cgmv_Envs_miss = get_missing_envs(data_Envs = cgmv_Envs)
# -

# Write out to be imputed envs / wholely absent envs
if [] != phno_Envs_miss:
    pd.DataFrame({'Absent_Envs':phno_Envs_miss}).to_csv('./data/Preparation/phno_Envs_miss.csv', index=False)
if [] != meta_Envs_miss:
    pd.DataFrame({'Absent_Envs':meta_Envs_miss}).to_csv('./data/Preparation/meta_Envs_miss.csv', index=False)
if [] != soil_Envs_miss:
    pd.DataFrame({'Absent_Envs':soil_Envs_miss}).to_csv('./data/Preparation/soil_Envs_miss.csv', index=False)
if [] != wthr_Envs_miss:
    pd.DataFrame({'Absent_Envs':wthr_Envs_miss}).to_csv('./data/Preparation/wthr_Envs_miss.csv', index=False)
if [] != cgmv_Envs_miss:
    pd.DataFrame({'Absent_Envs':cgmv_Envs_miss}).to_csv('./data/Preparation/cgmv_Envs_miss.csv', index=False)

assert [] == phno_Envs_miss
assert [] == meta_Envs_miss

add_envs = phno.loc[(phno.Env.isin(soil_Envs_miss)), ['Env', 'Year']].drop_duplicates()
soil = soil.merge(add_envs, how = 'outer')
soil_Envs_miss = get_missing_envs(data_Envs = list(set(soil['Env'])))
assert [] == soil_Envs_miss

add_envs = phno.loc[(phno.Env.isin(wthr_Envs_miss)), ['Env', 'Year']].drop_duplicates()
add_envs = add_envs.merge(
    wthr.loc[(wthr.Year.isin(list(set(add_envs['Year'])))), ['Year', 'Date']].drop_duplicates(),
    how = 'outer'
)
wthr = wthr.merge(add_envs, how = 'outer')
wthr_Envs_miss = get_missing_envs(data_Envs = list(set(wthr['Env'])))
assert [] == wthr_Envs_miss

add_envs = phno.loc[(phno.Env.isin(cgmv_Envs_miss)), ['Env', 'Year']].drop_duplicates()
cgmv = cgmv.merge(add_envs, how = 'outer')
cgmv_Envs_miss = get_missing_envs(data_Envs = list(set(cgmv['Env'])))
assert [] == cgmv_Envs_miss

# ## Reorganize columns

# +
# regroup phenotype and meta Columns
df = phno.merge(meta)

cols_for_phno = [
    'Env',
    'Year',
    'Hybrid',
    'Yield_Mg_ha',          #| <- Key target
    'Stand_Count_plants',   #|- Possible intermediate prediction targets
    'Pollen_DAP_days',      #|
    'Silk_DAP_days',        #|
    'Plant_Height_cm',      #|
    'Ear_Height_cm',        #|
    'Root_Lodging_plants',  #|
    'Stalk_Lodging_plants', #|
    'Grain_Moisture',       #|
    'Twt_kg_m3']            #|

cols_for_meta = [
    'Env',
    'Year',
    #  'Field_Location',
    #  'Experiment',
    #  'Replicate',
    #  'Block',
    #  'Plot',
    #  'Range',
    #  'Pass',
    'Hybrid',
    #  'Hybrid_orig_name',
    #  'Hybrid_Parent1',
    #  'Hybrid_Parent2',
    'Plot_Area_ha',     #|- Needs Imputation
    'Date_Planted',     #|
    'Date_Harvested',   #|
    'Experiment_Code',
    'Treatment',
    'City',
    'Farm',
    'Field',
    'Trial_ID (Assigned by collaborator for internal reference)', # 
    'Soil_Taxonomic_ID and horizon description, if known',        # 32.2% complete
    'Weather_Station_Serial_Number (Last four digits, e.g. m2700s#####)',
    'Weather_Station_Latitude (in decimal numbers NOT DMS)',
    'Weather_Station_Longitude (in decimal numbers NOT DMS)',
    #  'Date_weather_station_placed',
    #  'Date_weather_station_removed',
    'Previous_Crop',
    'Pre-plant_tillage_method(s)',
    'In-season_tillage_method(s)',
    'Type_of_planter (fluted cone; belt cone; air planter)',
    'System_Determining_Moisture',
    'Pounds_Needed_Soil_Moisture',
    'Latitude_of_Field_Corner_#1 (lower left)',
    'Longitude_of_Field_Corner_#1 (lower left)',
    'Latitude_of_Field_Corner_#2 (lower right)',
    'Longitude_of_Field_Corner_#2 (lower right)',
    'Latitude_of_Field_Corner_#3 (upper right)',
    'Longitude_of_Field_Corner_#3 (upper right)',
    'Latitude_of_Field_Corner_#4 (upper left)',
    'Longitude_of_Field_Corner_#4 (upper left)',
    'Cardinal_Heading_Pass_1',
    'Irrigated']

cols_for_cmnt =[
    'Env',
    'Year',
    'Issue/comment_#1',
    'Issue/comment_#2',
    'Issue/comment_#3',
    'Issue/comment_#4',
    'Issue/comment_#5',
    'Issue/comment_#6',
    'Comments']

phno = df.loc[:, cols_for_phno]
meta = df.loc[:, cols_for_meta]
cmnt = df.loc[:, cols_for_cmnt]

# +
# Deal with Issues/Comments
# I think I'll ignore these fields. There's qualitative information we can get
# and additional weather data (below) but not a lot that will be immediately 
# useful
cmnt = cmnt.drop_duplicates()
cmnt = cmnt.melt(id_vars = ['Env', 'Year'])
cmnt = cmnt.loc[(cmnt.value.notna()), :]
cmnt = cmnt.loc[(cmnt.value != 'nan'), :]

cmnts_2022  = list(set(list(cmnt.loc[cmnt.Year == "2022", 'value'])))
# cmnts_2022


# looking at comments from 2022 we could make use of some of these if we were 
# making manual guesses but these won't be too helpful here.
# - 'Some very low yields caused by greensnap\n',
# - 'Deer damage on the early hybrids',
# - 'sandhill cranes pulled out seedlings in sections of the field in late May/early June',
# - 'Thunderstorm caused very high level of root lodging and some greensnap.  Corn was upright at end of season, but mostly goosenecked.\nRoot lodging not recroded becuase of high percentage of goosenecking\nGreensnap not counted separately, so stalk lodging includes both goosenecking and stalk lodging\nSome very low yields caused by greensnap\n',

# However, the additional weather may be of use.
# TODO, get additional weather data
# 'Link to additional weather source available online: http://www.deos.udel.edu',
# 'Link to additional weather source available online: https://weather.cfaes.osu.edu//stationinfo.asp?id=13',
# 'Link to additional weather source available online: Georgia Weather - Automated Environmental Monitoring Network Page (uga.edu) - http://weather.uga.edu/mindex.php?content=calculator&variable=CC&site=TIFTON',
# 'Link to additional weather source available online: https://newa.cornell.edu/all-weather-data-query/   (select Aurora (CUAES Musgrave), NY)',
# 'Link to additional weather source available online: https://www.isws.illinois.edu/warm/stationmeta.asp?site=CMI&from=wx'
# more from pre 2022
# 'http://newa.cornell.edu/index.php?page=weather-station-page&WeatherStation=aur',
# 'Additional weather source available online: http://www.georgiaweather.net/?content=calculator&variable=CC&site=PENFIELD',
# 'http://www.deos.udel.edu',
# 'Additional weather data source: http://www.deos.udel.edu',
# ' https://soilseries.sc.egov.usda.gov/OSD_Docs/M/MEXICO.html',
# 'Link to additional weahter source: http://www.deos.udel.edu',
# 'Link to additional weahter source: Georgia Weather - Automated Environmental Monitoring Network Page (uga.edu) http://weather.uga.edu/mindex.php?variable=HI&site=TIFTON',
# 'Additional weather data for the farm can be found at: http://newa.cornell.edu/index.php?page=weather-station-page&WeatherStation=aur',
# 'Addicional weather source available online: http://agebb.missouri.edu/weather/history/index.asp?station_prefix=bfd',
# 'http://weather.uga.edu/mindex.php?content=calculator&variable=CC&site=TIFTON',


# [e for e in list(set(list(cmnt.loc[cmnt.Year != "2022", 'value']))) if e not in cmnts_2022]
# -

# ## Meta (1/2) Impute GPS Coordinates
#
#  Fix Missing GPS Coordinates.
#

# +
# Fix GPS coordinates
gps_cols = ['Latitude_of_Field_Corner_#1 (lower left)',
'Latitude_of_Field_Corner_#2 (lower right)',
'Latitude_of_Field_Corner_#3 (upper right)',
'Latitude_of_Field_Corner_#4 (upper left)',
            
'Longitude_of_Field_Corner_#1 (lower left)',
'Longitude_of_Field_Corner_#2 (lower right)',
'Longitude_of_Field_Corner_#3 (upper right)',
'Longitude_of_Field_Corner_#4 (upper left)',
            
'Weather_Station_Latitude (in decimal numbers NOT DMS)',
'Weather_Station_Longitude (in decimal numbers NOT DMS)']


gps = meta.loc[:, ['Env']+gps_cols]


# exclude sites outside of north america (side benefit of disqualifying wrongly input data from TXH1_2021)
# Logitude must be 
longitude_max = -60
latitude_max = 45

for col in gps_cols:
    print("Cleaning: '"+col+"'")
    if re.match('.*Latitude.+', col):
        mask = gps[col ]>latitude_max
    elif re.match('.*Longitude.+', col):
        mask = gps[col ]>longitude_max
    print("Erasing values in: '"+"', '".join(list(set(list(gps.loc[mask, 'Env']))))+"'")    
    gps.loc[mask, col] = np.nan
#     print('\n')

# +
# melt and get field center
gps = gps.melt(id_vars=['Env'])

mask = gps.variable.isin([
    'Latitude_of_Field_Corner_#1 (lower left)',
    'Latitude_of_Field_Corner_#2 (lower right)',
    'Latitude_of_Field_Corner_#3 (upper right)',
    'Latitude_of_Field_Corner_#4 (upper left)'
                  ])

gps.loc[mask, 'variable'] = 'Latitude_of_Field'

mask = gps.variable.isin([
    'Longitude_of_Field_Corner_#1 (lower left)',
    'Longitude_of_Field_Corner_#2 (lower right)',
    'Longitude_of_Field_Corner_#3 (upper right)',
    'Longitude_of_Field_Corner_#4 (upper left)'
                  ])

gps.loc[mask, 'variable'] = 'Longitude_of_Field'

# collapse measures for the field
gps = gps.groupby(['Env', 'variable']).agg(value = ('value', np.nanmean)).reset_index()

# if we have more accurate information, don't factor the weather station location into the estimate of the field location
gps['Replace'] = False
gps.loc[gps.value.isna(), 'Replace'] = True

mask = ((~(gps.Replace)) & (gps.variable == 'Weather_Station_Latitude (in decimal numbers NOT DMS)'))
gps.loc[mask, 'value'] = np.nan

mask = ((~(gps.Replace)) & (gps.variable == 'Weather_Station_Longitude (in decimal numbers NOT DMS)'))
gps.loc[mask, 'value'] = np.nan

gps = gps.drop(columns = ['Replace'])

# +
# repeat the same trick to use the weather station info if the field info is not known
mask = gps.variable.isin([
    'Weather_Station_Latitude (in decimal numbers NOT DMS)',
    'Latitude_of_Field'
                  ])

gps.loc[mask, 'variable'] = 'Latitude_of_Field'

mask = gps.variable.isin([
    'Weather_Station_Longitude (in decimal numbers NOT DMS)',
    'Longitude_of_Field'
                  ])

gps.loc[mask, 'variable'] = 'Longitude_of_Field'

gps = gps.groupby(['Env', 'variable']).agg(value = ('value', np.nanmean)).reset_index()

gps = gps.pivot(index='Env', columns='variable', values='value').reset_index()

# +
# Some envs have no gps info, impute those based on similarly named locations

# There is only one TXH4 group (TXH4_2019), so the below approach doesn't work. 
# I'll use all the TXH* sites to guess the coordinates.
mask = [True if re.match("TXH.+", e) else False for e in gps['Env']]

gps.loc[(gps.Env == 'TXH4_2019'), 'Latitude_of_Field'] = np.nanmedian(gps.loc[mask, 'Latitude_of_Field'])
gps.loc[(gps.Env == 'TXH4_2019'), 'Longitude_of_Field'] = np.nanmedian(gps.loc[mask, 'Longitude_of_Field'])

# Impute all non-TXH4 locations
gps[['EnvBase', 'EnvYear']] = gps.Env.str.split("_", expand = True)

mask_no_lat = gps.Latitude_of_Field.isna()
mask_no_lon = gps.Longitude_of_Field.isna()

impute_gps_vals = list(gps.loc[(mask_no_lat | mask_no_lon), 'Env'])


# for each Env with missing values, use the first portion of the name
# e.g. TXH1-Early_2017 -> TXH1 to search for possible matches
for impute_gps_val in impute_gps_vals:
    mask_gps_val = (gps.Env == impute_gps_val)
    match_root = gps.loc[mask_gps_val, 'EnvBase'].str.split('-')
    match_root = list(match_root)[0][0]
    #     print()

    mask = [True if re.match(e, match_root+'.+') else False for e in gps['EnvBase']]

    check_std_lat = round(np.nanstd(gps.loc[mask, 'Latitude_of_Field']), 3)
    check_std_lon = round(np.nanstd(gps.loc[mask, 'Longitude_of_Field']),3)

    gps.loc[mask_gps_val, 'Latitude_of_Field']  = np.nanmedian(gps.loc[mask, 'Latitude_of_Field'])
    gps.loc[mask_gps_val, 'Longitude_of_Field'] = np.nanmedian(gps.loc[mask, 'Longitude_of_Field'])

    
gps = gps.drop(columns = ['EnvBase', 'EnvYear'])
# -

assert sum(gps.Latitude_of_Field.isna()) == 0
assert sum(gps.Longitude_of_Field.isna()) == 0
print("All GPS Coordinates Imputed!")    
meta = meta.drop(columns = gps_cols).merge(gps)
print("And replaced in `meta`")

# ## Mock up CV Groupings

# +
# set(phno.Hybrid)
obs_summary = phno.loc[:, ['Env', 'Year', 'Hybrid', 'Yield_Mg_ha']].merge(meta.loc[:, ['Env', 'Latitude_of_Field', 'Longitude_of_Field']].drop_duplicates())

distance_threshold = 1 

# Create clusters with lat/lon
temp = obs_summary.loc[:, ['Env', 'Latitude_of_Field', 'Longitude_of_Field']
                      ].drop_duplicates(
                      ).reset_index(
                      ).drop(columns = 'index')
temp['GPS_Group'] = ""
temp['Distance'] = np.nan

gps_group_counter = 0

for i in temp.index:
    match_lat = temp.loc[i, 'Latitude_of_Field'] 
    match_lon = temp.loc[i, 'Longitude_of_Field'] 

    if temp.loc[i, 'GPS_Group'] == '':
        temp['Distance'] = np.sqrt( (temp['Latitude_of_Field'] - float(match_lat))**2 + (temp['Longitude_of_Field']  - float(match_lon))**2 )

        temp.loc[(temp.Distance <= distance_threshold), 'GPS_Group'] = str(gps_group_counter)
        gps_group_counter += 1


obs_summary = obs_summary.merge(temp.loc[:, ['Env', 'Latitude_of_Field', 'Longitude_of_Field', 'GPS_Group']])

# -

gps_groups_2022 = list(set(obs_summary.loc[(obs_summary.Year == '2022'), 'GPS_Group']))
mask_pre_2022 = (obs_summary.Year != '2022')
mask_gps_match = (obs_summary.GPS_Group.isin(gps_groups_2022))


def quick_obs_summary_tally(df = obs_summary.loc[:, ['Year', 'Env']]):
    df = df.groupby(['Year']
                   ).count(
                   ).reset_index(
                   ).assign(
        Pct   = lambda dataframe: round(dataframe['Env']/np.sum(dataframe['Env']), 4)*100,
        Total = lambda dataframe: np.sum(dataframe['Env']))
    return(df)


# Baseline: the whole dataset
quick_obs_summary_tally(df = obs_summary.loc[:, ['Year', 'Env']])

# The full possible testing dataset
obs_pre_2022 = quick_obs_summary_tally(df = obs_summary.loc[mask_pre_2022, ['Year', 'Env']])
obs_pre_2022

# +
# The testing dataset constrained to gps groups in 2022
obs_pre_2022_gps = quick_obs_summary_tally(df = obs_summary.loc[(mask_pre_2022 & mask_gps_match), ['Year', 'Env']])

obs_diff = obs_pre_2022_gps.loc[0, 'Total'] - obs_pre_2022.loc[0, 'Total']
obs_pct = 100*round(obs_diff/obs_pre_2022.loc[0, 'Total'], 4)
print(str(obs_diff)+' fewer obs\n'+str(obs_pct)+'% fewer obs')
obs_pre_2022_gps['Diff'] = obs_pre_2022_gps['Env']
obs_pre_2022_gps['Diff'] = obs_pre_2022_gps['Diff'] - obs_pre_2022['Env']

obs_pre_2022_gps
# -

# This doesn't remove too many observations from recent years but it's almost equivalent to a whole year. I think I'll not restrict the training set to only matching gps groups.
#
# Because I'll be ensembling models, I need multiple testing sets for
# 1. Hyperparameter selection
# 1.  
# 1. 
#
# ```
# Hyperparameter |-> Training |-> Ensemble |-> Predictions
# Selection      |            |   Tuning   |
# - 1 year       |            |            |
# ```

import itertools # for making test set permutations
testing_years = pd.DataFrame(
    [e for e in itertools.permutations([2014+i for i in range(8)], 3)], 
    columns = ['Test_HPS', 'Test_Model', 'Test_Ensemble'])


rng = np.random.default_rng(2039476435238045723476)
testing_years['Random_Order'] = rng.permutation(testing_years.shape[0])
testing_years.sort_values('Random_Order')

# ## Soil Drop Columns

# +
list(soil)

summarize_col_missing(soil)

# +
# drop low completion rate entries
temp = summarize_col_missing(df = soil)
# high percent complete columns
high_pr_comp_cols = list(temp.loc[(temp.Pr_Comp) > 50, # this threshold used to be 70, but the additon of envs 
                                  # which did not have soil measured deflate the completion rates
                                  'Col'])

soil = soil.loc[:, high_pr_comp_cols].drop(columns = [
    'LabID',           #|- Not interested in these columns
    'Date Received',   #|
    'Date Reported'])  #|


# -

# ## Meta Drop Columns

# write out a log of the enviroments imputed
def log_imputed_envs(
    df = meta,
    df_name = 'meta',
    col = 'Date_Planted'
):
    mask = df[col].isna()
    df = df.loc[mask, ['Env']].drop_duplicates().reset_index().drop(columns = ['index'])
    df.to_csv('./data/Preparation/'+df_name+'_Envs_imp_'+col+'.csv', index=False)  


# +
# discard columns that have low completion or redundant information
meta = meta.drop(columns = [e for e in list(meta) if e in [
    'Experiment_Code',
    'Treatment',
    'City',
    'Farm',
    'Field',
    'Trial_ID (Assigned by collaborator for internal reference)',
    'Soil_Taxonomic_ID and horizon description, if known',
    'Weather_Station_Serial_Number (Last four digits, e.g. m2700s#####)',
    'Type_of_planter (fluted cone; belt cone; air planter)',
    'In-season_tillage_method(s)', # 34% Pr_Comp
    'Plot_Area_ha', # 92.1 % complete but not a covariate I want to use
    'System_Determining_Moisture',
    'Cardinal_Heading_Pass_1',
    'Irrigated']])


summarize_col_missing(meta)
# -

# ### Previous Crop

log_imputed_envs(
    df = meta,
    df_name = 'meta',
    col = 'Previous_Crop'
) 

# need data dicts before I can encode
Previous_Crop = {
                                 'soybean': 'soy',
                                  'cotton': 'cotton',
                                   'wheat': 'wheat',
               'wheat/double crop soybean': 'soy_wheat',
                                    'corn': 'corn',
   'Lima beans followed by rye cover crop': 'lima_rye',
                                 'sorghum': 'sorghum',
                            'Winter wheat': 'wheat',
   'Fallow most of 2014 winter planted in fall of 2014 then sprayed with Glystar 24 floz/a on 5/3/15  and killed spring of 2015 spray ': 'fallow',
                                  'peanut': 'peanut',
           'wheat and Double Crop soybean': 'soy_wheat',
                              'sugar beet': 'beet',
    'Small Grains and Double Crop soybean': 'soy_rye',
                         'soybean/pumpkin': 'soy_pumpkin',
                           'wheat/soybean': 'soy_wheat',
     'soybeans with fall cereal rye cover': 'soy_rye'
}
set('_'.join(list(set([Previous_Crop[e] for e in Previous_Crop.keys()]))).split('_'))

# +
temp = pd.DataFrame(
    zip([e for e in Previous_Crop.keys()],
        [Previous_Crop[e] for e in Previous_Crop.keys()]), 
    columns = ['Previous_Crop', 'Value'])

for crop in ['beet', 'corn', 'cotton', 'fallow', 'lima', 
             'peanut', 'pumpkin', 'rye', 'sorghum', 'soy', 
             'wheat']:
    temp['Cover_'+crop]  = [1 if re.search(crop, e) else 0 for e in temp['Value']]
# -

meta = meta.merge(temp, how = 'outer').drop(columns = ['Value'])
meta = meta.drop(columns = 'Previous_Crop')

# ### Pre-plant_tillage_method(s)

log_imputed_envs(
    df = meta,
    df_name = 'meta',
    col = 'Pre-plant_tillage_method(s)'
) 

# +
Pre_plant_tillage_method = {
    'Conventional': 'Cult',
    'Disc in previous fall': 'Disc',
    'conventional': 'Cult',
    'field cultivator': 'Cult',
    'Fall Chisel': 'Chisel',
    'Fall chisel plow and spring field cultivate': 'Chisel_Cult',
    'chisel': 'Chisel',
    'No-till': 'None',
    'Chisel plow and field cultivator': 'Chisel_Cult',
    'chisel plow in fall; field cultivated in spring': 'Chisel_Cult',
    'In the Spring the land was cut with a disk, then ripped with a chisel plow to a depth of 8-10”. It was then cut again and we applied 300#/acre of 10-0-30-12%S. Next we used a field cultivator with rolling baskets to incorporate the fertilizer. The land was bedded just prior to planting.': 'Chisel_Cult_Disc',
    'no-till': 'None',
    'Field J was fall moldboard plow;  Then disked this spring and field cultivated before planting.': 'Chisel_Cult_Disc',
    'The field was minium tilled.  The field was disked then cultipacked then Cultimulched then planted': 'Cult_Disc_MinTill',
    'Fall Chisel Plow; Spring Cultivate': 'Chisel_Cult',
    'min-till': 'MinTill',
    'Field cultivator': 'Cult',
    'Field cultivate': 'Cult',
    'No-Till': 'None',
    'fall chisel plow, spring field cultivator': 'Chisel_Cult',
    'disc, conventional, followed by bedding': 'Disc_Cult',
    'No Till': 'None',
    'Chisel plowed 5/4/15 Disc and finishing tool 5/6/15 ': 'Chisel_Disc',
    'Chisel plowed 5/7/15 disc and field finisher 5/23/15': 'Chisel_Disc',
    'Fall Chisel, Spring Turbo-Till': 'Chisel_Cult',
    'Min-Till': 'MinTill',
    'Cultivate, hip and row': 'Cult',
    'Conventional disc tillage': 'Disc_Cult',
    'Field cultivated': 'Cult',
    'Chisel Plow': 'Chisel',
    'Conventional Tillage': 'Cult',
    'Moldboard plowed November\n': 'Chisel',
    'Fall Diskchisel, Spring Culivator': 'Disc_Cult',
    'Cultivator ': 'Cult',
    'Cultivate': 'Cult',
    'Min-Till ': 'MinTill',
    'Field Cultivator': 'Cult',
    'cultivate, hip, row': 'Cult',
    'none': 'None',
    'Chisel': 'Chisel',
    'Fall plow/Spring field cultivator': 'Chisel_Cult',
    'disk and hip': 'Disc',
    'till and hip': 'Chisel',
    '1 pass with soil finisher': 'Cult',
    'disked, chisel plow and field cultivator': 'Chisel_Disc_Cult',
    'harrowed, rototilled': 'Cult',
    'disc': 'Disc',
    'Two passes with disk, one pass with field conditioner, 30Ó beds were made with 8 row ripper-bedder for corn': 'Disc_Cult',
    'Fall Diskchisel / Spring Culivator': 'Chisel_Cult',
    'chisel plow': 'Chisel',
    '2 passes with a field cultivator': 'Cult',
    'disk': 'Disc',
    'Chisel plow,cultivate': 'Chisel_Cult',
    'Fall soil chisel, spring cultivator': 'Chisel_Cult',
    'field cultivate': 'Cult',
    'cultivate, hip and row': 'Cult',
    'disked, ripped, field cultivator': 'Disc_Cult',
    'Ripper Bed, rototill': 'Chisel',
    'standard (disk plow)': 'Disc_Chisel_Cult',
    'spring field cultivator': 'Cult',
    'fall disk chisel, spring cultivator': 'Chisel_Cult',
    'chisel, field cultivate': 'Chisel_Cult',
    'Chisel plow followed by  cultivator': 'Chisel_Cult',
    'field cultivate (twice)': 'Cult',
    'Chisel plow': 'Chisel',
    'Chisel plowed on 12/14/17': 'Chisel',
    'Heavy disk, Chisel plow, Field cultivator': 'Chisel_Disc_Cult',
    'Disked, chisil, disk': 'Chisel_Disc_Cult',
    'Ripper Bed, Rototill': 'Cult',
    'disked and field conditioned': 'Disc',
    'field cultivate ': 'Cult',
    'Case IH 335 VT 4" deep ': 'Cult',
    'Chisel - field cultivator': 'Cult',
    'Chisel plow followed by cultivator': 'Chisel_Cult',
    'CaseIH VT 360 vertical tillage tool gone over 2X on 5/22': 'Cult',
    'CaseIH VT 360 vertical tillage tool gone over 2X on 5/23': 'Cult',
    'Fall Diskchisel, Spring Disk': 'Disc',
    'Cultivator': 'Cult',
    'disc harrow followed by chisel plow, field cultivator used to prepare final seed bed': 'Disc_Chisel_Cult',
    'harrow, ripper bed, rototill': 'Cult',
    'soil finisher': 'None',
    'conventional, heavy disc, chisel plow, field cultivator': 'Disc_Chisel_Cult',
    'cultivate 2x': 'Cult',
    'cultivator': 'Cult',
    'Conventional tillage (disc) + ripped and bedded rows': 'Disc',
    'Heavy Disk and then rows placed': 'Disc',
    'Chisel plow, disk, field cultivator': 'Disc_Chisel_Cult',
    'Harrow, ripper bed, rototill': 'Cult',
    'Field cultivator. Tillage with soil finisher': 'Cult',
    'Strip tillage': 'Cult',
    'Fall rip one pass tool, spring field cultivate': 'Cult',
    'conventional - Field Cultivator, disk': 'Cult',
    'None': 'None',
    'CaseIH VT 360 vertical tillage tool gone over 2X on 5/18/2021': 'Cult',
    'CaseIH VT 360 vertical tillage tool gone over 2X on 5/18': 'Cult',
    'Conventional, heavy disc, chisel plow, field cultivator': 'Disc_Chisel_Cult',
    'Disked the whole field and then rows were placed': 'Disc',
    'Disc, Dynadrive': 'Disc',
    'ripper bed, rototill': 'Cult',
    'Disk': 'Disc',
    'Disk, field cultivator, ripper/bedder': 'Disc_Chisel_Cult',
    'Fall disk chisel, Spring cultivator': 'Disc_Chisel_Cult',
    'conventional, heavy disc, chisel plow, and field cultivator': 'Disc_Chisel_Cult',
    'Discing': 'Disc_Chisel_Cult'
}

set('_'.join(list(set([Pre_plant_tillage_method[e] for e in Pre_plant_tillage_method.keys()]))).split('_'))

# +
temp = pd.DataFrame(
    zip([e for e in Pre_plant_tillage_method.keys()],
        [Pre_plant_tillage_method[e] for e in Pre_plant_tillage_method.keys()]), 
    columns = ['Pre-plant_tillage_method(s)', 'Value'])

temp['Pre_Chisel']  = [1 if re.search('Chisel', e) else 0 for e in temp['Value']]
temp['Pre_Cult']    = [1 if re.search('Cult', e) else 0 for e in temp['Value']]
temp['Pre_Disc']    = [1 if re.search('Disc', e) else 0 for e in temp['Value']]
temp['Pre_MinTill'] = [1 if re.search('MinTill', e) else 0 for e in temp['Value']]

temp
# -

meta = meta.merge(temp, how = 'outer').drop(columns = ['Pre-plant_tillage_method(s)', 'Value'])

# ### Pounds_Needed_Soil_Moisture

log_imputed_envs(
    df = meta,
    df_name = 'meta',
    col = 'Pounds_Needed_Soil_Moisture'
) 

# +
meta = sanitize_col(df = meta, col = 'Pounds_Needed_Soil_Moisture', 
             simple_renames= {
                 'Unknown': '-9999',
                 'Unknown, currently getting in contact with manufacturer, technician estimated about 5 lbs of grain to get moisture reading':'5',
 '4 or 5':'4.5',
 '~2.5':'2.5',
 "Based on the technitian's experience a minimum of 4 lbs is required. However the user manual says the minimum volume for accurate determination is 2 liters":'4',
 'Depend on moisture content of grain 15.5% moisture 5.84 lbs 30% moisture 7.05 lbs':'6.445',
 '2.5 lbs.':'2.5',
 '5-6.5':'5.75',
 '3 to 4':'3.5',
 '~5 lbs':'5',
 '~10 lbs':'10',
 '7 to 9':'8',
 '<1': '0.5'
             }, 
             split_renames= {})

meta['Pounds_Needed_Soil_Moisture'] =  meta.loc[:, 'Pounds_Needed_Soil_Moisture'].astype(float)

mask = meta['Pounds_Needed_Soil_Moisture'] == -9999
meta.loc[mask, 'Pounds_Needed_Soil_Moisture'] = np.nan
# -

# ### Numerical imputation

# +
pre_plant_cols = [
    'Pre_Chisel',
    'Pre_Cult',
    'Pre_Disc',
    'Pre_MinTill']

cover_crop_cols = [
    'Cover_beet',
    'Cover_corn',
    'Cover_cotton',
    'Cover_fallow',
    'Cover_lima',
    'Cover_peanut',
    'Cover_pumpkin',
    'Cover_rye',
    'Cover_sorghum',
    'Cover_soy',
    'Cover_wheat']

# +
# mode impute the one hot encoded variables
from scipy import stats # for stats.mode for imputation

for col in pre_plant_cols+cover_crop_cols:
    temp = meta.loc[:, ['Env', col]].drop_duplicates()
    imp_val = stats.mode(temp[col], keepdims = False).mode
    mask = meta[col].isna()
    meta.loc[mask, col] = imp_val


# +
# median impute soil moisture
mask = meta['Pounds_Needed_Soil_Moisture'].isna()

meta.loc[mask, 'Pounds_Needed_Soil_Moisture'] = np.nanmedian((meta['Pounds_Needed_Soil_Moisture']))
# -

# ## Soil Impute Missing Values

# +
# Impute soil based on GPS coordinates
# Use distance between lon/lat to fill in missing values with the nearest. 
# I'm using euclidean distance of lon/lat so this is not perfect.
temp = meta.loc[:, ['Env', 'Latitude_of_Field', 'Longitude_of_Field']].drop_duplicates()

soil = soil.merge(temp).drop(columns = 'Texture') # with sand silt clay percents this is redundant info

check_cols = [e for e in list(soil) if e not in ['Env', 'Year', 'Latitude_of_Field', 'Longitude_of_Field']]

for check_col in check_cols:
    # check_col = check_cols[0]
    mask = soil[check_col].isna()

    fill_envs = soil.loc[mask, 'Env']
    for fill_env in fill_envs:
        # fill_env = fill_envs[0]
        
        if np.dtype(soil[check_col]) != 'O':
            match_lat = soil.loc[(soil.Env == fill_env),  'Latitude_of_Field'] 
            match_lon = soil.loc[(soil.Env == fill_env),  'Longitude_of_Field'] 

            soil['Distance'] = np.sqrt( ((soil['Latitude_of_Field']  - float(match_lat))**2
                                    ) + ((soil['Longitude_of_Field'] - float(match_lon))**2))

            dist_min = np.nanmin(soil.loc[(soil.Env != fill_env), 'Distance'])

#             print(dist_min)
            dist_mask = soil.Distance == dist_min

            soil.loc[(soil.Env == fill_env), check_col] = np.nanmedian(soil.loc[dist_mask, check_col])

soil = soil.drop(columns = ['Latitude_of_Field', 'Longitude_of_Field', 'Distance'])
# -

# Impute soil without GPS coordinates
imputer = KNNImputer(n_neighbors=3)
knn_imputed = pd.DataFrame(
    imputer.fit_transform(
        soil.drop(columns = ['Env', 'Year'])))

# +
knn_imputed.columns = [e for e in list(soil) if e not in ['Env', 'Year']]

soil = pd.concat([soil.loc[:, ['Env', 'Year']], knn_imputed], axis = 1)
# -

# on WSL this was fine, but on linux it is asserting an error desipte there being none below 100.0. Below works just fine.
# assert False not in (summarize_col_missing(soil).loc[:, 'Pr_Comp'] == 100) 
# Below works as expected.
assert False not in [True if e == 100 else False for e in list(summarize_col_missing(soil).loc[:, 'Pr_Comp'])]
print("No missing values in `soil`")


# ## Weather Impute Missing

# +
def dl_power_data(
    latitude = 32.929, 
    longitude = -95.770,
    start_YYYYMMDD = 20150101,
    end_YYYYMMDD = 20150305
):
    # Modified by 
    # https://power.larc.nasa.gov/docs/tutorials/service-data-request/api/
    '''
    *Version: 2.0 Published: 2021/03/09* Source: [NASA POWER](https://power.larc.nasa.gov/)
    POWER API Multi-Point Download
    This is an overview of the process to request data from multiple data points from the POWER API.
    '''

    base_url = r"https://power.larc.nasa.gov/api/temporal/daily/point?parameters=QV2M,T2MDEW,PS,RH2M,WS2M,GWETTOP,ALLSKY_SFC_SW_DWN,ALLSKY_SFC_PAR_TOT,T2M_MAX,T2M_MIN,T2MWET,GWETROOT,T2M,GWETPROF,ALLSKY_SFC_SW_DNI,PRECTOTCORR&community=RE&longitude={longitude}&latitude={latitude}&start={start_YYYYMMDD}&end={end_YYYYMMDD}&format=JSON"

    api_request_url = base_url.format(
        longitude=longitude, 
        latitude=latitude,
        start_YYYYMMDD=start_YYYYMMDD, 
        end_YYYYMMDD=end_YYYYMMDD)

    response = requests.get(url=api_request_url, verify=True, timeout=30.00)

    content = json.loads(response.content.decode('utf-8'))

    # Repackage content as data frame
    df_list = [
        pd.DataFrame(content['properties']['parameter'][e], index = [0]).melt(
        ).rename(columns = {'variable':'Date', 'value':e})
        for e in list(content['properties']['parameter'].keys())
    ]

    for i in range(len(df_list)):
        if i == 0:
            out = df_list[i]
        else:
            out = out.merge(df_list[i])

    out['Latitude'] = latitude
    out['Longitude'] = longitude
    first_cols = ['Latitude', 'Longitude', 'Date']
    out = out.loc[:, first_cols+[e for e in list(out) if e not in first_cols]]
    return(out)


# dl_power_data(
#     latitude = 32.929, 
#     longitude = -95.770,
#     start_YYYYMMDD = 20150101,
#     end_YYYYMMDD = 20150305
# )

# for reference, here's some info on the structure of contents
# dict_keys(['type', 'geometry', 'properties', 'header', 'messages', 'parameters', 'times'])
# content['header']
# {'title': 'NASA/POWER CERES/MERRA2 Native Resolution Daily Data',
#  'api': {'version': 'v2.3.5', 'name': 'POWER Daily API'},
#  'sources': ['power', 'ceres', 'merra2'],
#  'fill_value': -999.0,
#  'start': '20150101',
#  'end': '20150305'}
# content['parameters']
# {'QV2M': {'units': 'g/kg', 'longname': 'Specific Humidity at 2 Meters'},
#  'T2MDEW': {'units': 'C', 'longname': 'Dew/Frost Point at 2 Meters'},
#  'PS': {'units': 'kPa', 'longname': 'Surface Pressure'},
#  'RH2M': {'units': '%', 'longname': 'Relative Humidity at 2 Meters'},
#  'WS2M': {'units': 'm/s', 'longname': 'Wind Speed at 2 Meters'},
#  'GWETTOP': {'units': '1', 'longname': 'Surface Soil Wetness'},
#  'ALLSKY_SFC_SW_DWN': {'units': 'kW-hr/m^2/day',
#   'longname': 'All Sky Surface Shortwave Downward Irradiance'},
#  'ALLSKY_SFC_PAR_TOT': {'units': 'W/m^2',
#   'longname': 'All Sky Surface PAR Total'},
#  'T2M_MAX': {'units': 'C', 'longname': 'Temperature at 2 Meters Maximum'},
#  'T2M_MIN': {'units': 'C', 'longname': 'Temperature at 2 Meters Minimum'},
#  'T2MWET': {'units': 'C', 'longname': 'Wet Bulb Temperature at 2 Meters'},
#  'GWETROOT': {'units': '1', 'longname': 'Root Zone Soil Wetness'},
#  'T2M': {'units': 'C', 'longname': 'Temperature at 2 Meters'},
#  'GWETPROF': {'units': '1', 'longname': 'Profile Soil Moisture'},
#  'ALLSKY_SFC_SW_DNI': {'units': 'kW-hr/m^2/day',
#   'longname': 'All Sky Surface Shortwave Downward Direct Normal Irradiance'},
#  'PRECTOTCORR': {'units': 'mm/day', 'longname': 'Precipitation Corrected'}}

# +
# this shows that the downloaded values and given values are almost in perfect
# agreement (ALLSKY_SFC_SW_DWN, ALLSKY_SFC_SW_DNI) are the only discrepencies 
# for the test case. I'll list out those sites with errors and selectively 
# download them and fill in missing values. To be polite, check if the weather
# data already exists before drawing from POWER's API.

# NOTE. Most, but not all of the missing values are due to the release date. 
# -

if True == False:
    wthr_DEH1_2014 = wthr.loc[wthr.Env == 'DEH1_2014', :]
    lat, lon = list(meta.loc[meta.Env == 'DEH1_2014', ['Latitude_of_Field', 'Longitude_of_Field']].loc[0, :])

    powr_DEH1_2014 = dl_power_data(
        latitude = lat, 
        longitude = lon,
        start_YYYYMMDD = np.min(wthr_DEH1_2014.Date),
        end_YYYYMMDD = np.max(wthr_DEH1_2014.Date)
    )

    pd.DataFrame(
        [(e, (np.mean(wthr_DEH1_2014[e] - powr_DEH1_2014[e]))
         ) for e in list(wthr_DEH1_2014) if e not in ['Env', 'Year', 'Date']
        ], columns = ['Measure', 'Total_Difference']
    )

# +
polite_request_interval = 10 # time in seconds

# check if clean weather exists, load if it does, download and fill in if not

if 'wthr_powr_imp.csv' in os.listdir('./data/Preparation/'):
    print('Reading Weather from Preparation/')
    wthr = pd.read_csv('./data/Preparation/wthr_powr_imp.csv')
else:
    print('Filling in Weather from NASA Power')
    # Figure out what Envs need to be imputed
    temp = wthr

    temp = temp.drop(columns = ['Year', 'Date'])
    for col in [e for e in list(temp) if e != 'Env']:
        temp[col] = temp[col].isna()
    temp = temp.drop_duplicates()

    temp['Num_Rep'] = temp.drop(columns = ['Env']).sum(axis = 1)
    impute_envs = list(temp.loc[(temp.Num_Rep > 0), 'Env'].drop_duplicates())
    # impute_envs
    
    
    for impute_env in impute_envs:
        # impute_env = 'WIH3_2022'#impute_envs[0]

        meta_mask = (meta.Env == impute_env)
        lat = float(meta.loc[meta_mask, 'Latitude_of_Field'].drop_duplicates())
        lon = float(meta.loc[meta_mask, 'Longitude_of_Field'].drop_duplicates())

        wthr_mask = (wthr.Env == impute_env)

        date_min = np.min(wthr.loc[wthr_mask, 'Date'])
        date_max = np.max(wthr.loc[wthr_mask, 'Date'])

        # Delay between requests 
        if impute_env != impute_envs[0]:
            time.sleep(polite_request_interval)

        powr_dl = dl_power_data(
            latitude = lat, 
            longitude = lon,
            start_YYYYMMDD = date_min,
            end_YYYYMMDD = date_max
        )

        # cut out the previous download and add in the next
        # change type to allow for merging
        powr_dl['Date'] = powr_dl['Date'].astype(int)
        wthr_slice = wthr.loc[wthr_mask, ['Env', 'Year', 'Date']].merge(powr_dl)
        wthr = wthr.loc[~wthr_mask, :].merge(wthr_slice, how = 'outer')
        
    wthr = wthr.drop(columns = ['Latitude', 'Longitude'])
    wthr.to_csv('./data/Preparation/wthr_powr_imp.csv', index=False)
# -

# check if clean weather exists, load if it does, download and fill in if not
if 'wthr_powr_imp_knn.csv' in os.listdir('./data/Preparation/'):
    wthr_knn = pd.read_csv('./data/Preparation/wthr_powr_imp_knn.csv')
else:    
    # For values that are missing in POWER (-999), remove and then knn impute them
    for col in [e for e in list(wthr) if sum((wthr[e] == -999)) > 0 ]:
        mask = (wthr[col] == -999)
        wthr.loc[mask, col] = np.nan    
    
    
    imputer = KNNImputer(n_neighbors=20)
    knn_imputed = pd.DataFrame(
        imputer.fit_transform(
            wthr.drop(columns = ['Env', 'Year', 'Date'])))

    knn_imputed.columns = [e for e in list(wthr) if e not in ['Env', 'Year', 'Date']]

    wthr_knn = pd.concat([wthr.loc[:, ['Env', 'Year', 'Date']], knn_imputed], axis = 1)
    wthr_knn.to_csv('./data/Preparation/wthr_powr_imp_knn.csv', index=False)


# +
#clip to last doy
wthr = wthr_knn

temp = wthr.loc[:, ['Year', 'Date']].drop_duplicates()
temp['Date_Str'] = temp['Date'].astype(str)
temp['DOY'] = [pd.Period(e, freq='D').day_of_year for e in list(temp['Date'])]


# 2022 is the constraint. It only goes up to day 314. 
clip_doy = np.max(temp.loc[(temp.Year == 2022), 'DOY'])

temp = temp.loc[(temp.DOY <= clip_doy), ]
temp = temp.drop(columns = ['Date_Str'])

wthr = wthr.merge(temp)
# -

# same problem as above. Unclear why. 
# assert False not in (summarize_col_missing(wthr).loc[:, 'Pr_Comp'] == 100)
assert False not in [True if e == 100 else False for e in list(summarize_col_missing(wthr).loc[:, 'Pr_Comp'])]
print("No missing values in `wthr`")

# ## CGMV Impute Missing

log_imputed_envs(
    df = cgmv,
    df_name = 'cgmv',
    col = 'SDR_pGerEme_1'
) 

mask = cgmv.SDR_pGerEme_1.isna()
imp_Envs = cgmv.loc[mask, ['Env', ]].drop_duplicates()
imp_Envs = imp_Envs.reset_index().drop(columns = 'index')


gps_lookup = meta.loc[:, ['Env', 'Latitude_of_Field', 'Longitude_of_Field']].drop_duplicates()
# get rid of sites that need to be imputed in cgmv so the closest value is good to go. 
antimask = gps_lookup.Env.isin(imp_Envs['Env'])
gps_search = gps_lookup.loc[antimask, ]
gps_lookup = gps_lookup.loc[~antimask, ]
gps_lookup['Distance'] = np.nan


def temp_find_closes(Env):
    mask = (gps_search.Env == Env)
    lat = list(gps_search.loc[mask, ]['Latitude_of_Field'])[0]
    lon = list(gps_search.loc[mask, ]['Longitude_of_Field'])[0]

    gps_lookup['Distance'] = np.sqrt( ((gps_lookup['Latitude_of_Field']  - float(lat))**2
                                        ) + ((gps_lookup['Longitude_of_Field'] - float(lon))**2))

    out = gps_lookup.loc[(gps_lookup.Distance == min(gps_lookup.Distance)), 'Env']
    return(list(out))


imp_with_Envs = [temp_find_closes(e) for e in list(imp_Envs.Env)]

# this is very sloppy but it's effective and fast to write
# for all the envs that need to be matched, get the closest Env(s) then loop
# over the cols
for i in tqdm.tqdm(range(len(list(imp_Envs.Env)))):
    fillin = list(imp_Envs.Env)[i]
    fillinwith = imp_with_Envs[i]

    mask_fillin = cgmv.Env == fillin
    mask_fillinwith = cgmv.Env.isin(fillinwith)

    for col in [e for e in list(cgmv) if e not in ['Env', 'Year']]:
        fillin_value = np.nanmean(cgmv.loc[mask_fillinwith, col])
        cgmv.loc[mask_fillin, col] = fillin_value


summarize_col_missing(cgmv)

# # Test Model Workflow: Impute Missing Management Data 

# +
# Test out the workflow in miniature by imputing Planting date
mask = (testing_years.Random_Order == 66)
# Test_HPS, Test_Model, Test_Ensemble =
holdout_years = list(
    testing_years.loc[mask, [
        'Test_HPS', 
        'Test_Model', 
        'Test_Ensemble']].reset_index().drop(columns = 'index').loc[0, :])

holdout_years = holdout_years + [2022]
holdout_years = [str(e) for e in holdout_years]


# 1. Hyperparameters =====
hps_hold = holdout_years[-4:]
hps_test = holdout_years[0:1]

# 2. Model =====
mod_hold = holdout_years[-3:]
mod_test = holdout_years[1:2]

# 3. Ensemble =====
ens_hold = holdout_years[-2:]
ens_test = holdout_years[2:3]

# 4. Submission =====
sub_hold = holdout_years[-1:]
sub_test = holdout_years[3:4]


# -

# Making the input data 
class df_prep():
    def __init__(self):
        self.train = {
            "set":None,
              "x":None,
              "y":None,
            "yna":None
        }    
        self.test = {
            "set":None,
              "x":None,
              "y":None,
            "yna":None
        }  
        self.cs_dict = None
        self.isolate_missing_y_run = False # This is just for a guard rail in mk_scale_dict
       
    # set up the dfs for the y var
    def get_train_test_Envs(
        self,
        df = meta,
        holdout_years = ['2020', '2014', '2016', '2022'],
        test_year = ['2020']
        ):
            mask = df.Year.isin(holdout_years+test_year)
            train_set = df.loc[~mask, ['Env', 'Year']].drop_duplicates()

            mask = df.Year.isin(test_year)
            test_set = df.loc[mask, ['Env', 'Year']].drop_duplicates()
            
            self.train['set'] = train_set
            self.test['set'] = test_set

    ## Retrieve data based on Envs in test/train sets ==========================
    # add in y variable
    def _mk_ys_df(
        self,
        df_envs, #= train_HPS, # self. ['set']
        df_data = meta,
        add_cols = ['Date_Planted']
        ):
        df_data = df_data.loc[:, ['Env']+add_cols]
        df_out = df_envs.merge(df_data, 'left').drop_duplicates()
        return(df_out)
    
    def add_ys(
        self,
        #df_envs = train_HPS,
        df_data = meta,
        add_cols = ['Date_Planted']
        ):
        self.train['y'] = self._mk_ys_df(
            df_envs = self.train['set'],
            df_data = df_data,
            add_cols = add_cols)
        
        self.test['y'] = self._mk_ys_df(
            df_envs = self.test['set'],
            df_data = df_data,
            add_cols = add_cols)
        
    #  add in x variables
    def _mk_xs_df(
        self,
        df_envs, #= train_HPS,
        df_data = wthr,
        drop_cols = ['Year', 'Date', 'DOY']
        ):    
        df_data = df_data.drop(columns = [e for e in list(df_data) if e in drop_cols])
        df_out = df_envs.merge(df_data, 'left').drop_duplicates()
        df_out = df_out.drop(columns = [e for e in list(df_out) if e in drop_cols])
        return(df_out)
    
    def add_xs(
        self,
        #df_envs = train_HPS,
        df_data = wthr,
        drop_cols = ['Year', 'Date', 'DOY']
        ):
        self.train['x'] = self._mk_xs_df(
            df_envs = self.train['set'],
            df_data = df_data,
            drop_cols = drop_cols)

        self.test['x'] = self._mk_xs_df(
            df_envs = self.test['set'],
            df_data = df_data,
            drop_cols = drop_cols)
    
    ## Separate out missing responses ==========================================
    def isolate_missing_y(self):
        if self.train['y'] is not None:
            mask = (self.train['y'].isnull().any(axis = 1))
            self.train['yna'] = self.train['y'].loc[mask, :].copy()
            self.train['y'] = self.train['y'].loc[~mask, :].copy()
        if self.test['y'] is not None:
            mask = (self.test['y'].isnull().any(axis = 1))
            self.test['yna'] = self.test['y'].loc[mask, :].copy()
            self.test['y'] = self.test['y'].loc[~mask, :].copy()
            
        self.isolate_missing_y_run = True  # This is just for a guard rail in mk_scale_dict
            
    def prep_idx_y(self):
        if self.train['y'] is not None:
            self.train['y'] = self.train['y'].reset_index().drop(columns = 'index')
        if self.train['yna'] is not None:
            self.train['yna'] = self.train['yna'].reset_index().drop(columns = 'index')
            
        if self.test['y'] is not None:
            self.test['y'] = self.test['y'].reset_index().drop(columns = 'index') 
        if self.test['yna'] is not None:
            self.test['yna'] = self.test['yna'].reset_index().drop(columns = 'index')
            
    ## Center &  Scaling dict ==================================================
    # This looks odd (why not only have one method?) but is intentional. The idea here is that one
    # might want to provide a saved center and scaling dictionary or include custom scaling for some 
    # columns. By including an update method making the default to return the dictionary instead of 
    # updating it silently it's easier to access and makes the step more visible. 
    def update_cs_dict(self, cs_dict):
        self.cs_dict = cs_dict            
    
    def mk_scale_dict(self, 
                     scale_cols = ['Date_Planted'],
                     return_cs_dict = True,
                     store_cs_dict = False):
        # scale df
        if not self.isolate_missing_y_run: 
            print("Warning: if run before isolate_missing_y all observations will be used.")
            
        temp = self.train['y'].merge(self.train['x'], how = 'left')
        cs_dict = {}
        for e in scale_cols:
            cs_dict.update({e : {'mean': np.mean(temp[e]), 
                                 'std' : np.std(temp[e])}})
            
        if store_cs_dict:
            if self.cs_dict is not None:
                print('Overwriting Center and Scaling Dict.')
            self.update_cs_dict(cs_dict)   
            
        if return_cs_dict:
            return(cs_dict) 

    ## Scaling / reverse scaling by dict =======================================
    def _scale_by_dict(self, df):
        scale_cols = [e for e in list(self.cs_dict.keys()) if e in list(df)]
        for e in scale_cols:
            df[e] = (df[e] - self.cs_dict[e]['mean'])/self.cs_dict[e]['std']
        return(df)
    
    def _unscale_by_dict(self, df):
        scale_cols = [e for e in list(self.cs_dict.keys()) if e in list(df)]
        for e in scale_cols:
            df[e] = (df[e]*self.cs_dict[e]['std'])+self.cs_dict[e]['mean']
        return(df)
            
    def apply_scaling(self):
        if self.train[ 'y'] is not None:
            self.train['y']   = self._scale_by_dict(self.train['y'])
        if self.train[ 'yna'] is not None:
            self.train['yna'] = self._scale_by_dict(self.train['yna'])
        if self.train[ 'x'] is not None:
            self.train['x']   = self._scale_by_dict(self.train['x'])            
            
        if self.test[ 'y'] is not None:
            self.test['y']   = self._scale_by_dict(self.test['y']) 
        if self.test[ 'yna'] is not None:
            self.test['yna'] = self._scale_by_dict(self.test['yna'])        
        if self.test[ 'x'] is not None:
            self.test['x']   = self._scale_by_dict(self.test['x']) 
        
    def reverse_scaling(self):
        if self.train[ 'y'] is not None:
            self.train['y']   = self._unscale_by_dict(self.train['y'])
        if self.train[ 'yna'] is not None:
            self.train['yna'] = self._unscale_by_dict(self.train['yna'])
        if self.train[ 'x'] is not None:
            self.train['x']   = self._unscale_by_dict(self.train['x'])
            
        if self.test[ 'y'] is not None:
            self.test['y']   = self._unscale_by_dict(self.test['y']) 
        if self.test[ 'yna'] is not None:
            self.test['yna'] = self._unscale_by_dict(self.test['yna']) 
        if self.test[ 'x'] is not None:
            self.test['x']   = self._unscale_by_dict(self.test['x'])
      
    
    ## numpy arrays to easily be converted to tensors ==========================
    def mk_arrays(
        self,
        split = 'train',
        obs_per_Env = 1, 
        return_2d   = True,
        missing_ys  = False):
        # using the envs specified in the y data frame and the covariates in the x data frame, return numpy arrays 
        # with the xs and ys accessible by the same idx 

        def _reformat_xy(y_df, 
                         x_df, 
                         obs_per_Env, 
                         return_2d = False):
            ys_tensor = np.array(y_df.loc[:, [e for e in y_df if e not in ['Env', 'Year']]])
            # Don't seem to need to manually set second dim (to 1 from none) do this when data is drawn from df asabove
            # ys_tensor = ys_tensor.reshape((ys_df.shape[0], 1))

            col_names = [e for e in list(x_df) if e not in ['Env']]

            num_obs = ys_tensor.shape[0] # observations
            # obs_per_Env # user input, 1 in most cases, 314 for weather
            col_per_obs = len(col_names)

            xs_tensor = np.zeros(shape = (num_obs,
                                          obs_per_Env,
                                          col_per_obs)) 

            for i in y_df.index:
                mask = (x_df.Env == y_df.loc[i, 'Env'])
                xs_tensor[i, :, :] = np.array(x_df.loc[mask, col_names
                                                       ].drop_duplicates() )
                # Note! this will result in weather data being in order of 
                # N, Length (days), Channels. To match pytorch conventions 
                # I swap these below
                
            # for non weather data
            if return_2d:
                new_dim_0 = xs_tensor.shape[0]
                new_dim_1 = xs_tensor.shape[2]
                xs_tensor = xs_tensor.reshape(new_dim_0, new_dim_1)
            else:
                # swap axes so that weather is in order of 
                # N, Channels, Length
                xs_tensor = xs_tensor.swapaxes(1,2)

            return([ys_tensor, xs_tensor])

        if split not in ['train', 'test']:
            print('`split`must be "train" or "test"')
        else:
            if split == 'train':
                if not missing_ys:
                    out = _reformat_xy(
                        y_df = self.train['y'], 
                        x_df = self.train['x'], 
                        obs_per_Env = obs_per_Env, 
                        return_2d = return_2d)
                else:
                    out = _reformat_xy(
                        y_df = self.train['yna'], 
                        x_df = self.train['x'], 
                        obs_per_Env = obs_per_Env, 
                        return_2d = return_2d)

            elif split == 'test':
                if not missing_ys:
                    out = _reformat_xy(
                        y_df = self.test['y'], 
                        x_df = self.test['x'], 
                        obs_per_Env = obs_per_Env, 
                        return_2d = return_2d)
                else:
                    out = _reformat_xy(
                        y_df = self.test['yna'], 
                        x_df = self.test['x'], 
                        obs_per_Env = obs_per_Env, 
                        return_2d = return_2d)
            return(out)


# Convert dates to doy so they're easier to work with
def df_date_to_datetime(df, cols):
    temp = df.copy()
    for col in cols:
        temp[col] = [pd.Period(e, freq='D').day_of_year for e in list(temp[col])]
    return(temp)


# Demo 3d data (wthr)
if True == False:
    demo = df_prep()
    demo.get_train_test_Envs(
        df = meta, 
        holdout_years = ['2020', '2014', '2016', '2022'],
        test_year =     ['2020'] )
    demo.add_ys(
            df_data = df_date_to_datetime(df = meta, 
                                          cols = ['Date_Planted']),
            add_cols = ['Date_Planted'])
    demo.add_xs(
            df_data = wthr,
            drop_cols = ['Year', 'Date', 'DOY'])
    demo.isolate_missing_y()
    demo.prep_idx_y()
    demo.mk_scale_dict(
        scale_cols = ['Date_Planted']+[e for e in list(wthr) if e not in ['Env', 'Year', 'Date', 'DOY']],
        return_cs_dict = False,
        store_cs_dict = True)
    demo.apply_scaling()
    # demo.reverse_scaling()

    train_y, train_x = demo.mk_arrays(
        split = 'train',
        obs_per_Env = 314, 
        return_2d   = False,
        missing_ys  = False)

    test_y, test_x = demo.mk_arrays(
        split = 'test',
        obs_per_Env = 314, 
        return_2d   = False,
        missing_ys  = False)

    print([e.shape for e in [train_y, train_x, test_y, test_x]])

# Demo 2d data (soil)
if True == False:
    demo = df_prep()
    demo.get_train_test_Envs(
        df = meta, 
        holdout_years = ['2020', '2014', '2016', '2022'],
        test_year =     ['2020'] )
    demo.add_ys(
            df_data = df_date_to_datetime(df = meta, 
                                          cols = ['Date_Planted']),
            add_cols = ['Date_Planted'])
    demo.add_xs(
            df_data = soil, 
            drop_cols = ['Year', 'Date', 'DOY'])
    demo.isolate_missing_y()
    demo.prep_idx_y()
    demo.mk_scale_dict(
        scale_cols = ['Date_Planted']+[e for e in list(soil) if e not in ['Env', 'Year', 'Date', 'DOY']],
        return_cs_dict = False,
        store_cs_dict = True)
    demo.apply_scaling()
    # demo.reverse_scaling()

    train_y, train_x = demo.mk_arrays(
        split = 'train',
        obs_per_Env = 1, 
        return_2d   = True,
        missing_ys  = False)

    test_y, test_x = demo.mk_arrays(
        split = 'test',
        obs_per_Env = 1, 
        return_2d   = True,
        missing_ys  = False)


    print([e.shape for e in [train_y, train_x, test_y, test_x]])


# Making the input data
class CustomDataset_wthr(Dataset):
    def __init__(self, y, x, 
                 transform=None, # can pass in ToTensor()
                 target_transform=None):
        self.y = y
        self.x = x
        self.transform = transform
        self.target_transform = target_transform    
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        x_idx = self.x[idx]
        y_idx = self.y[idx]

        if self.transform:
            x_idx = self.transform(x_idx)
        if self.target_transform:
            y_idx = self.target_transform(y_idx)
        return x_idx, y_idx


# +
def train_loop(dataloader, model, loss_fn, optimizer, silent = True):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            if not silent:
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn, silent = True):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
#     correct /= size
    if not silent:
        print(f"Test Error: Avg loss: {test_loss:>8f}")
    return(test_loss) #new
# -

# ## Shared Data Prep

# +
# # Demo 3d data (wthr)

# demo = df_prep()
# demo.get_train_test_Envs(df = meta, 
#                          holdout_years = ['2020', '2014', '2016', '2022'], 
#                          test_year =     ['2020'] )
# demo.add_ys(df_data = df_date_to_datetime(df = meta, cols = ['Date_Planted']),
#             add_cols = ['Date_Planted'])
# demo.add_xs(df_data = wthr, 
#             drop_cols = ['Year', 'Date', 'DOY'])
# demo.isolate_missing_y()
# demo.prep_idx_y()
# demo.mk_scale_dict(
#     scale_cols = ['Date_Planted']+[e for e in list(wthr) if e not in ['Env', 'Year', 'Date', 'DOY']],
#     return_cs_dict = False,
#     store_cs_dict = True)
# demo.apply_scaling()

# train_y, train_x = demo.mk_arrays(
#     split = 'train',
#     obs_per_Env = 314, 
#     return_2d   = False,
#     missing_ys  = False)

# test_y, test_x = demo.mk_arrays(
#     split = 'test',
#     obs_per_Env = 314, 
#     return_2d   = False,
#     missing_ys  = False)

# [e.shape for e in [train_y, train_x, test_y, test_x]]
# -

# ## PyTorch Models

# +
# train_y_tensor = torch.from_numpy(train_y).to(device).float()
# train_x_tensor = torch.from_numpy(train_x).to(device).float()

# test_y_tensor = torch.from_numpy(test_y).to(device).float()
# test_x_tensor = torch.from_numpy(test_x).to(device).float()

# training_dataloader = DataLoader(CustomDataset_wthr(y = train_y_tensor, x = train_x_tensor), batch_size = 64, shuffle = True)
# testing_dataloader = DataLoader(CustomDataset_wthr(y = test_y_tensor, x = test_x_tensor), batch_size = 64, shuffle = True)


# class NeuralNetwork(nn.Module):
#     def __init__(self):
#         super(NeuralNetwork, self).__init__()    
#         in_size = 314 * 16
#         n1 = 16 
#         n2 = 16 
    
#         self.linear_relu_stack = nn.Sequential(
#         nn.Flatten(),            
#         nn.Linear(in_size, n1),
#         nn.ReLU(),            
#         nn.Linear(n1, n2),
#         nn.ReLU(),            
#         nn.Linear(n2, 1)
#         )

#     def forward(self, x):
#         # x = self.flatten(x)
#         logits = self.linear_relu_stack(x)
#         return logits
    
    
# model = NeuralNetwork().to(device)
# # print(model)
# # print(model(torch.rand(1,  device=device))) # shows that it works
# # model(next(iter(training_dataloader))[0] ) # try prediction on one batch



# # learning_rate = 1e-3
# # batch_size = 64
# # epochs = 500
# #
# # # Initialize the loss function
# # loss_fn = nn.MSELoss()
# # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# #
# # loss_df = pd.DataFrame([i for i in range(epochs)], columns = ['Epoch'])
# # loss_df['MSE'] = np.nan
# #
# # for t in tqdm.tqdm(range(epochs)):
# #     # print(f"Epoch {t+1}\n-------------------------------")
# #     train_loop(training_dataloader, model, loss_fn, optimizer)
# #    
# #     loss_df.loc[loss_df.index == t, 'MSE'
# #                ] = test_loop(testing_dataloader, model, loss_fn)
# #
# # print("Done!")



# def train_nn(
#     training_dataloader,
#     testing_dataloader,
#     model,
#     learning_rate = 1e-3,
#     batch_size = 64,
#     epochs = 500
# ):
#     # Initialize the loss function
#     loss_fn = nn.MSELoss()
#     optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#     loss_df = pd.DataFrame([i for i in range(epochs)], columns = ['Epoch'])
#     loss_df['MSE'] = np.nan

#     for t in tqdm.tqdm(range(epochs)):
#         # print(f"Epoch {t+1}\n-------------------------------")
#         train_loop(training_dataloader, model, loss_fn, optimizer)

#         loss_df.loc[loss_df.index == t, 'MSE'
#                    ] = test_loop(testing_dataloader, model, loss_fn)

#     # print("Done!")
#     return([model, loss_df])


# model, loss_df = train_nn(
#     training_dataloader,
#     testing_dataloader,
#     model,
#     learning_rate = 1e-3,
#     batch_size = 64,
#     epochs = 500
# )


# yhats = model(test_x_tensor)
# yhats = yhats.cpu().detach().numpy()

# yobs = test_y_tensor
# yobs = yobs.cpu().detach().numpy()

# plt_df = pd.concat([
#     pd.DataFrame(yhats, columns = ['yHat']),
#     pd.DataFrame(yobs, columns = ['yObs'])], 
#     axis=1)

# +
# px.line(loss_df, x = 'Epoch', y = 'MSE')

# +
# px.scatter(plt_df, x = 'yObs', y = 'yHat')

# +
# mean_squared_error(plt_df['yObs'], plt_df['yHat'], squared=False)
# -

# ### Simple dense

# +
test_this_year = '2021'
# Setup ----------------------------------------------------------------------
trial_name = 'DNN_hps_test'

learning_rate = 1e-3
batch_size = 64
epochs = 500

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
    
        in_size = 314 * 16
        n1 = 2**10
        n2 = 2**8
        n3 = 2**6
        n4 = 2**4
        n5 = 2 
    
        self.linear_relu_stack = nn.Sequential(
        nn.Flatten(),            
        nn.Linear(in_size, n1),
        nn.ReLU(),            
        nn.Linear(n1, n2),
        nn.ReLU(),            
        nn.Linear(n2, n3),
        nn.ReLU(),            
        nn.Linear(n3, n4),
        nn.ReLU(),            
        nn.Linear(n4, n5),
        nn.ReLU(),
        nn.Linear(n5, 1)
        )

    def forward(self, x):
#         x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def run_dnn_trial(
    trial_name = 'DNN_hps_test',
    learning_rate = 1e-3,
    batch_size = 64,
    epochs = 500):
    reset_trial_name = trial_name
    for test_this_year in ['2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021']:
        trial_name = reset_trial_name
        trial_name = trial_name+test_this_year
        print(test_this_year)
        # Data Prep. -----------------------------------------------------------------
        data_obj = df_prep()
        data_obj.get_train_test_Envs(df = meta, 
                                 holdout_years = [],
                                 test_year =     [test_this_year] )
        data_obj.add_ys(df_data = df_date_to_datetime(df = meta, cols = ['Date_Planted']),
                    add_cols = ['Date_Planted'])
        data_obj.add_xs(df_data = wthr, 
                    drop_cols = ['Year', 'Date', 'DOY'])
        data_obj.isolate_missing_y()
        data_obj.prep_idx_y()
        data_obj.mk_scale_dict(
            scale_cols = ['Date_Planted']+[e for e in list(wthr) if e not in ['Env', 'Year', 'Date', 'DOY']],
            return_cs_dict = False,
            store_cs_dict = True)
        data_obj.apply_scaling()

        train_y, train_x = data_obj.mk_arrays(
            split = 'train',
            obs_per_Env = 314, 
            return_2d   = False,
            missing_ys  = False)

        test_y, test_x = data_obj.mk_arrays(
            split = 'test',
            obs_per_Env = 314, 
            return_2d   = False,
            missing_ys  = False)


        train_y_tensor = torch.from_numpy(train_y).to(device).float()
        train_x_tensor = torch.from_numpy(train_x).to(device).float()

        test_y_tensor = torch.from_numpy(test_y).to(device).float()
        test_x_tensor = torch.from_numpy(test_x).to(device).float()


        training_dataloader = DataLoader(CustomDataset_wthr(y = train_y_tensor, x = train_x_tensor), batch_size = 64, shuffle = True)
        testing_dataloader = DataLoader(CustomDataset_wthr(y = test_y_tensor, x = test_x_tensor), batch_size = 64, shuffle = True)

        # Fit Mod --------------------------------------------------------------------   
        cache_save_name = cache_path+trial_name+'_mod.pth'
        if os.path.exists(cache_save_name):
            model = torch.load(cache_save_name)
            model = NeuralNetwork().to(device)

        else:
            model = NeuralNetwork().to(device)
            def train_nn(
                training_dataloader,
                testing_dataloader,
                model,
                learning_rate = 1e-3,
                batch_size = 64,
                epochs = 500
            ):
                # Initialize the loss function
                loss_fn = nn.MSELoss()
                optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
                loss_df = pd.DataFrame([i for i in range(epochs)], columns = ['Epoch'])
                loss_df['MSE'] = np.nan

                for t in tqdm.tqdm(range(epochs)):
                    # print(f"Epoch {t+1}\n-------------------------------")
                    train_loop(training_dataloader, model, loss_fn, optimizer)

                    loss_df.loc[loss_df.index == t, 'MSE'
                               ] = test_loop(testing_dataloader, model, loss_fn)
                return([model, loss_df])


            model, loss_df = train_nn(
                training_dataloader,
                testing_dataloader,
                model,
                learning_rate = learning_rate,
                batch_size = batch_size,
                epochs = epochs
            )
            # Save
            torch.save(model, cache_save_name)
            loss_df.to_csv(cache_path+trial_name+'_loss.csv', index = False)


        # Eval. Best HPS -------------------------------------------------------------    
        for i in range(4):
            # u denotes that the true value is unknown
            temp_label= ['train_y', 'test_y', 'train_u', 'test_u'][i]
            if os.path.exists(cache_path+trial_name+temp_label+".csv"):
                pass
            else:
                temp_data = [data_obj.train['y'], data_obj.test['y'], data_obj.train['yna'], data_obj.test['yna']][i]

                if temp_data.shape[0] != 0:
                    # calculation step below =============================================
                    temp_y, temp_x = data_obj.mk_arrays(                                 #
                        split = temp_label.split('_')[0],                                #
                        obs_per_Env = 314,                                               #
                        return_2d   = False,                                             #
                        missing_ys  = temp_label.split('_')[1] == 'u' )                  #
                                                                                         #
                    temp_yHat = model(torch.from_numpy(temp_x).to(device).float())       #
                    temp_yHat = temp_yHat.cpu().detach().numpy()                         #
                    temp_yHat.reshape(temp_yHat.shape[0], ) # collapse to 1d #
                    # calculation step above =============================================
                    temp_data['yHat'] = [e[0] for e in list(temp_yHat)]
                    temp_data.to_csv(cache_path+trial_name+'_'+temp_label+".csv", index = False)

                if temp_label.split('_')[1] != 'u':
                    pd.DataFrame({'MSE':[mean_squared_error(temp_y, temp_yHat)]}).to_csv(cache_path+trial_name+'_'+temp_label+"_mse.csv", index = False)
# -

if False:
    run_dnn_trial(
        trial_name = 'DNN_hps_test',
        learning_rate = 1e-3,
        batch_size = 64,
        epochs = 500)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
    
        in_size = 314 * 16
        n1 = 2**10
        n4 = 2**4
        n5 = 2 
    
        self.linear_relu_stack = nn.Sequential(
        nn.Flatten(),            
        nn.Linear(in_size, n1),
        nn.Linear(n1, n4),
        nn.ReLU(),            
        nn.Linear(n4, n5),
        nn.ReLU(),
        nn.Linear(n5, 1)
        )

    def forward(self, x):
#         x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
if False:
    run_dnn_trial(
        trial_name = 'DNNsmall_hps_test',
        learning_rate = 1e-3,
        batch_size = 64,
        epochs = 500)


# +
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()  
        self.seq_model = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size = 3, stride=2), 
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size = 3, stride=2), 
            nn.ReLU(),
            nn.Conv1d(16, 8, kernel_size = 3, stride=2),       
            nn.ReLU(),
            nn.Conv1d(8, 4, kernel_size = 3, stride=2),       
            nn.ReLU(),
            nn.Conv1d(4, 2, kernel_size = 3, stride=2),       
            nn.ReLU(),
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        # x = self.flatten(x)
        logits = self.seq_model(x)
        return logits
    
if False:       
    run_dnn_trial(
        trial_name = 'CNNv1_hps_test',
        learning_rate = 1e-3,
        batch_size = 64,
        epochs = 500)

# +
# code to help with checking dims

# test_x_ten = torch.from_numpy(test_x).float()
# test_x_ten.shape

# #          Out shape
# #      channels    |
# #             |    |
# m = nn.Sequential(
#     nn.Conv1d(16, 32, kernel_size = 3, stride=2), 
#     nn.ReLU(),
#     nn.Conv1d(32, 16, kernel_size = 3, stride=2), 
#     nn.ReLU(),
#     nn.Conv1d(16, 8, kernel_size = 3, stride=2),       
#     nn.ReLU(),
#     nn.Conv1d(8, 4, kernel_size = 3, stride=2),       
#     nn.ReLU(),
#     nn.Conv1d(4, 2, kernel_size = 3, stride=2),       
#     nn.ReLU(),
#     nn.Flatten(),
#     nn.ReLU(),
#     nn.Linear(16, 1)
# )

# m(test_x_ten).shape
# -

# ### Reflect on DNNs

# +
loss_files = [e for e in os.listdir(cache_path) if re.match('.+_loss.csv$', e)]

loss_dfs = [pd.read_csv(cache_path+e) for e in loss_files]
for i in range(len(loss_files)):
    loss_dfs[i]['File'] = loss_files[i]
    
loss_df = pd.concat(loss_dfs)
# -

loss_df[['Model', 'Stage', 'CV', 'Record']] = loss_df['File'].str.split('_', expand = True)
loss_df['CV'] = loss_df['CV'].str.strip('test')

px.line(loss_df, x = 'Epoch', y = 'MSE', color = 'CV', facet_col="Model")


# ## Sklearn Model

# +
# transform to panel data
def wthr_rank_3to2(x_3d):
    n_obs, n_days, n_metrics = x_3d.shape
    return(x_3d.reshape(n_obs, (n_days*n_metrics)))

def y_rank_2to1(y_2d):
    n_obs = y_2d.shape[0]
    return(y_2d.reshape(n_obs, ))


# -

# ### Random Forest

# +
# train_x_2d = wthr_rank_3to2(x_3d = train_x)
# train_y_1d = y_rank_2to1(y_2d = train_y)
# regr = RandomForestRegressor(max_depth= 16, 
#                              random_state=0,
#                              n_estimators = 20)
# rf = regr.fit(train_x_2d, train_y_1d)
# mean_squared_error(train_y_1d, rf.predict(train_x_2d), squared=False)
# px.bar(pd.DataFrame(dict(cols=trn_xs.columns, imp=rf.feature_importances_)), x = 'cols', y = 'imp')

# +
test_this_year = '2021'

# Setup ----------------------------------------------------------------------
trial_name = 'rf_hps_test'

n_trials= 200 
n_jobs = 20

def objective(trial):
    rf_max_depth = trial.suggest_int('rf_max_depth', 2, 200, log=True)
    rf_n_estimators = trial.suggest_int('rf_n_estimators', 2, 200, log=True)
    rf_min_samples_split = trial.suggest_float('rf_min_samples_split', 0.01, 0.99, log=True)
    
    regr = RandomForestRegressor(
        max_depth = rf_max_depth, 
        n_estimators = rf_n_estimators,
        min_samples_split = rf_min_samples_split
        )
    
    rf = regr.fit(train_x_2d, train_y_1d)
    return (mean_squared_error(train_y_1d, rf.predict(train_x_2d), squared=False))


reset_trial_name = trial_name

if False:
    for test_this_year in ['2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021']:
        trial_name = reset_trial_name
        trial_name = trial_name+test_this_year
        print(test_this_year)
        # Data Prep. -----------------------------------------------------------------
        data_obj = df_prep()
        data_obj.get_train_test_Envs(df = meta, 
                                 holdout_years = [],
                                 test_year =     [test_this_year] )
        data_obj.add_ys(df_data = df_date_to_datetime(df = meta, cols = ['Date_Planted']),
                    add_cols = ['Date_Planted'])
        data_obj.add_xs(df_data = wthr, 
                    drop_cols = ['Year', 'Date', 'DOY'])
        data_obj.isolate_missing_y()
        data_obj.prep_idx_y()
        data_obj.mk_scale_dict(
            scale_cols = ['Date_Planted']+[e for e in list(wthr) if e not in ['Env', 'Year', 'Date', 'DOY']],
            return_cs_dict = False,
            store_cs_dict = True)
        data_obj.apply_scaling()

        train_y, train_x = data_obj.mk_arrays(
            split = 'train',
            obs_per_Env = 314, 
            return_2d   = False,
            missing_ys  = False)

        test_y, test_x = data_obj.mk_arrays(
            split = 'test',
            obs_per_Env = 314, 
            return_2d   = False,
            missing_ys  = False)


        train_x_2d = wthr_rank_3to2(x_3d = train_x)
        train_y_1d = y_rank_2to1(y_2d = train_y)


        # HPS Study ------------------------------------------------------------------
        cache_save_name = cache_path+trial_name+'_hps.pkl'
        if os.path.exists(cache_save_name):
            study = pkl.load(open(cache_save_name, 'rb'))  
        else:
            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials= n_trials, n_jobs = n_jobs)
            # save    
            pkl.dump(study, open(cache_save_name, 'wb'))    


        # Fit Best HPS ---------------------------------------------------------------   
        cache_save_name = cache_path+trial_name+'_mod.pkl'
        if os.path.exists(cache_save_name):
            rf = pkl.load(open(cache_save_name, 'rb'))  
        else:
            regr = RandomForestRegressor(
                    max_depth = study.best_trial.params['rf_max_depth'], 
                    n_estimators = study.best_trial.params['rf_n_estimators'],
                    min_samples_split = study.best_trial.params['rf_min_samples_split']
                    )
            rf = regr.fit(train_x_2d, train_y_1d)
            # save    
            pkl.dump(rf, open(cache_save_name, 'wb'))   


            # Eval. Best HPS -------------------------------------------------------------    
            for i in range(4):
                # u denotes that the true value is unknown
                temp_label= ['train_y', 'test_y', 'train_u', 'test_u'][i]
                if os.path.exists(cache_path+trial_name+temp_label+".csv"):
                    pass
                else:
                    temp_data = [data_obj.train['y'], data_obj.test['y'], data_obj.train['yna'], data_obj.test['yna']][i]

                    if temp_data.shape[0] != 0:
                        # calculation step below =============================================
                        temp_y, temp_x = data_obj.mk_arrays(                                 #
                            split = temp_label.split('_')[0],                                #
                            obs_per_Env = 314,                                               #
                            return_2d   = False,                                             #
                            missing_ys  = temp_label.split('_')[1] == 'u' )                  #
                                                                                             #
                        temp_y = y_rank_2to1(y_2d = temp_y)                                  #
                        temp_yHat = rf.predict(wthr_rank_3to2(x_3d = temp_x))                #
                        # calculation step above =============================================
                        temp_data['yHat'] = list(temp_yHat)
                        temp_data.to_csv(cache_path+trial_name+'_'+temp_label+".csv", index = False)

                        if temp_label.split('_')[1] != 'u':
                            pd.DataFrame({'MSE':[mean_squared_error(temp_y, temp_yHat)]}).to_csv(cache_path+trial_name+'_'+temp_label+"_mse.csv", index = False)
# -

# ### XGBoost

from xgboost import XGBRegressor

# +
test_this_year = '2021'

# Setup ----------------------------------------------------------------------
trial_name = 'xgb_hps_test'

n_trials= 200 # FIXME
n_jobs = 20

def objective(trial):
    xgb_max_depth = trial.suggest_int('xgb_max_depth', 2, 200, log=True)
    xgb_n_estimators = trial.suggest_int('xgb_n_estimators', 2, 200, log=True)
    xgb_learning_rate = trial.suggest_float('xgb_learning_rate', 0.0001, 0.3, log=True)
    
    regr = XGBRegressor(
        max_depth = xgb_max_depth, 
        n_estimators = xgb_n_estimators,
        learning_rate = xgb_learning_rate,
        objective='reg:squarederror'
        )
    
    xgb = regr.fit(train_x_2d, train_y_1d)
    return (mean_squared_error(train_y_1d, xgb.predict(train_x_2d), squared=False))


reset_trial_name = trial_name

if False:
    for test_this_year in ['2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021']:
        trial_name = reset_trial_name
        trial_name = trial_name+test_this_year
        print(test_this_year)
        # Data Prep. -----------------------------------------------------------------
        data_obj = df_prep()
        data_obj.get_train_test_Envs(df = meta, 
                                 holdout_years = [],
                                 test_year =     [test_this_year] )
        data_obj.add_ys(df_data = df_date_to_datetime(df = meta, cols = ['Date_Planted']),
                    add_cols = ['Date_Planted'])
        data_obj.add_xs(df_data = wthr, 
                    drop_cols = ['Year', 'Date', 'DOY'])
        data_obj.isolate_missing_y()
        data_obj.prep_idx_y()
        data_obj.mk_scale_dict(
            scale_cols = ['Date_Planted']+[e for e in list(wthr) if e not in ['Env', 'Year', 'Date', 'DOY']],
            return_cs_dict = False,
            store_cs_dict = True)
        data_obj.apply_scaling()

        train_y, train_x = data_obj.mk_arrays(
            split = 'train',
            obs_per_Env = 314, 
            return_2d   = False,
            missing_ys  = False)

        test_y, test_x = data_obj.mk_arrays(
            split = 'test',
            obs_per_Env = 314, 
            return_2d   = False,
            missing_ys  = False)


        train_x_2d = wthr_rank_3to2(x_3d = train_x)
        train_y_1d = y_rank_2to1(y_2d = train_y)


        # HPS Study ------------------------------------------------------------------
        cache_save_name = cache_path+trial_name+'_hps.pkl'
        if os.path.exists(cache_save_name):
            study = pkl.load(open(cache_save_name, 'rb'))  
        else:
            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials= n_trials, n_jobs = n_jobs)
            # save    
            pkl.dump(study, open(cache_save_name, 'wb'))    


        # Fit Best HPS ---------------------------------------------------------------   
        cache_save_name = cache_path+trial_name+'_mod.pkl'
        if os.path.exists(cache_save_name):
            xgb = pkl.load(open(cache_save_name, 'rb'))  
        else:
            regr = XGBRegressor(
                    max_depth = study.best_trial.params['xgb_max_depth'], 
                    n_estimators = study.best_trial.params['xgb_n_estimators'],
                    learning_rate = study.best_trial.params['xgb_learning_rate'],
                    objective='reg:squarederror'
                    )

            xgb = regr.fit(train_x_2d, train_y_1d)
            # save    
            pkl.dump(xgb, open(cache_save_name, 'wb'))   

            # Eval. Best HPS -------------------------------------------------------------    
            for i in range(4):
                # u denotes that the true value is unknown
                temp_label= ['train_y', 'test_y', 'train_u', 'test_u'][i]
                if os.path.exists(cache_path+trial_name+temp_label+".csv"):
                    pass
                else:
                    temp_data = [data_obj.train['y'], data_obj.test['y'], data_obj.train['yna'], data_obj.test['yna']][i]

                    if temp_data.shape[0] != 0:
                        # calculation step below =============================================
                        temp_y, temp_x = data_obj.mk_arrays(                                 #
                            split = temp_label.split('_')[0],                                #
                            obs_per_Env = 314,                                               #
                            return_2d   = False,                                             #
                            missing_ys  = temp_label.split('_')[1] == 'u' )                  #
                                                                                             #
                        temp_y = y_rank_2to1(y_2d = temp_y)                                  #
                        temp_yHat = xgb.predict(wthr_rank_3to2(x_3d = temp_x))               #
                        # calculation step above =============================================
                        temp_data['yHat'] = list(temp_yHat)
                        temp_data.to_csv(cache_path+trial_name+'_'+temp_label+".csv", index = False)

                        if temp_label.split('_')[1] != 'u':
                            pd.DataFrame({'MSE':[mean_squared_error(temp_y, temp_yHat)]}).to_csv(cache_path+trial_name+'_'+temp_label+"_mse.csv", index = False)
# -

# ## Aggregate Estimates
# Test out inverse variance weighting

# +
# Find all errors
mse_files = [e for e in os.listdir(cache_path) if re.match('.+_mse.csv$', e)]
mse_files = [e for e in mse_files if re.match('.+hps.+', e)]

mod_mses = [pd.read_csv(cache_path+e) for e in mse_files]
for i in range(len(mse_files)):
    mod_mses[i]['File'] = mse_files[i]
    
mod_mse = pd.concat(mod_mses)

mod_mse[['Model', 'Stage', 'CV', 'Set', 'Discard1', 'Discard2']] = mod_mse['File'].str.split('_', expand = True)
mod_mse = mod_mse.drop(columns = ['Discard1', 'Discard2'])
mod_mse['CV'] = mod_mse['CV'].str.strip('test')
# -

mod_mse.loc[mod_mse.Model == 'rf', ]

px.scatter(mod_mse, x = "Model", y = 'MSE', color = 'CV')

# +
test_files = [e for e in os.listdir(cache_path) if re.match('.+_test_y.csv$', e)]
test_files = [e for e in test_files if re.match('.+hps.+', e)]

test_preds = [pd.read_csv(cache_path+e) for e in test_files]
for i in range(len(test_files)):
    test_preds[i]['File'] = test_files[i]
    
test_pred = pd.concat(test_preds)

test_pred[['Model', 'Stage', 'CV', 'Set', 'Discard1']] = test_pred['File'].str.split('_', expand = True)
test_pred = test_pred.drop(columns = ['Discard1'])
test_pred['CV'] = test_pred['CV'].str.strip('test')

test_pred
# -

test_pred_wide = test_pred.pivot(columns='Model', values='yHat', index = ['Env', 'Year', 'CV', 'Date_Planted'])
test_pred_wide = test_pred_wide.reset_index()
test_pred_wide

px.scatter_matrix(test_pred_wide, dimensions=['Date_Planted', 'CNNv1', 'DNN', 'DNNsmall', 'rf', 'xgb'], 
                 color = 'CV')

# +
mod_list = [e for e in list(test_pred_wide) if e not in ['Env', 'Year', 'CV', 'Date_Planted']]
var_list = [(test_pred_wide['Date_Planted'] - test_pred_wide[e]).var() for e in mod_list]
wht_list = [e/np.sum(var_list) for e in var_list]


# Aggregate estimates using averaging and inverse variance weighting
for i in range(len(mod_list)):
    if i == 0:
        yHat_ave_accumulator = test_pred_wide[mod_list[i]]*(1/len(mod_list))
    else:
        yHat_ave_accumulator += test_pred_wide[mod_list[i]]*(1/len(mod_list))

for i in range(len(mod_list)):
    if i == 0:
        yHat_invVar_accumulator = test_pred_wide[mod_list[i]]*wht_list[i]
    else:
        yHat_invVar_accumulator += test_pred_wide[mod_list[i]]*wht_list[i]


# -

test_pred_wide['mean'] = yHat_ave_accumulator
test_pred_wide['iVar'] = yHat_invVar_accumulator

test_pred_long = test_pred_wide.melt(id_vars=['Env', 'Year', 'CV', 'Date_Planted'])
test_pred_long['ErrorSq'] = (test_pred_long['Date_Planted'] - test_pred_long['value'])**2
test_pred_long = test_pred_long.groupby(['CV', 'Model']).agg(MSE = ('ErrorSq', np.mean)).reset_index()

test_pred_long.head()

px.box(test_pred_long, x = 'Model', y = 'MSE', color = 'Model')

# ## Impute Missing

# write out a log of the enviroments imputed
log_imputed_envs(
    df = meta,
    df_name = 'meta',
    col = 'Date_Planted'
)   


# +
# get scaling to use on predictions
def temp_fcn(test_this_year):
    demo = df_prep()
    demo.get_train_test_Envs(df = meta, 
                             holdout_years = [], 
                             test_year =     [test_this_year] )
    demo.add_ys(df_data = df_date_to_datetime(df = meta, cols = ['Date_Planted']),
                add_cols = ['Date_Planted'])
    demo.add_xs(df_data = wthr, 
                drop_cols = ['Year', 'Date', 'DOY'])
    demo.isolate_missing_y()
    demo.prep_idx_y()
    out = demo.mk_scale_dict(
        scale_cols = ['Date_Planted'],
        return_cs_dict = True,
        store_cs_dict = False)
    return(out)

years_tested = ['2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021']
# -

yHat_scaling = [pd.DataFrame(temp_fcn(test_this_year)['Date_Planted'], index = [test_this_year]) for test_this_year in years_tested]
yHat_scaling = pd.concat(yHat_scaling).reset_index().rename(columns = {'index':'CV'})
yHat_scaling.head()

# +
impute_with_mod = 'rf'

file_list = [e for e in os.listdir(cache_path) if re.match('^'+impute_with_mod+'.+.csv$', e) and not re.match('.+mse.csv$', e)]
# allow for all to further model stacking
# for now I'm just going to use one model
#file_list = [e for e in file_list if not  re.match('.+train_y.csv$', e)]

table_list = [pd.read_csv(cache_path+e) for e in file_list]
for i in range(len(file_list)):
    table_list[i]['File'] = file_list[i]
    
yHat = pd.concat(table_list)

yHat[['Model', 'Stage', 'CV', 'Set', 'Value_Known']] = yHat['File'].str.split('_', expand = True)
yHat['CV'] = yHat['CV'].str.strip('test')
yHat.head()

# +
yHat = yHat.merge(yHat_scaling)

yHat['yHat'] = (yHat['yHat']*yHat['std'])+yHat['mean']

yHat = yHat.loc[:, ['Env', 'Year', 'Model', 'CV', 'yHat']]
# TODO if using multiple models and weights, those should be applied here

yHat= yHat.groupby(['Env']).agg(yHat = ('yHat', np.nanmean)).reset_index()
yHat.head()

# +
# Impute missing values
temp = meta

temp = temp.loc[:, ['Year', 'Date_Planted']].drop_duplicates()
temp['Date_Str'] = temp['Date_Planted'].astype(str)
temp['DOY'] = [pd.Period(e, freq='D').day_of_year for e in list(temp['Date_Str'])]

temp = meta.merge(temp).merge(yHat)

px.scatter(temp, 'DOY', 'yHat')

# +
mask = (temp.DOY.isna())
temp.loc[mask, 'DOY'] = temp.loc[mask, 'yHat']
temp = temp.drop(columns = ['Date_Str', 'yHat', 'Date_Planted']).rename(columns = {'DOY': 'Date_Planted'})

meta = temp
# -

# # Apply RF Imputation for missing Date_Harvested

summarize_col_missing(meta)

# +
# Setup ----------------------------------------------------------------------
trial_name = 'rf_impHarvest_test'

n_trials= 200 
n_jobs = 20

def objective(trial):
    rf_max_depth = trial.suggest_int('rf_max_depth', 2, 200, log=True)
    rf_n_estimators = trial.suggest_int('rf_n_estimators', 2, 200, log=True)
    rf_min_samples_split = trial.suggest_float('rf_min_samples_split', 0.01, 0.99, log=True)
    
    regr = RandomForestRegressor(
        max_depth = rf_max_depth, 
        n_estimators = rf_n_estimators,
        min_samples_split = rf_min_samples_split
        )
    
    rf = regr.fit(train_x_2d, train_y_1d)
    return (mean_squared_error(train_y_1d, rf.predict(train_x_2d), squared=False))


reset_trial_name = trial_name

if False:
    for test_this_year in ['2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021']:
        trial_name = reset_trial_name
        trial_name = trial_name+test_this_year
        print(test_this_year)
        # Data Prep. -----------------------------------------------------------------
        data_obj = df_prep()
        data_obj.get_train_test_Envs(df = meta, 
                                 holdout_years = [],
                                 test_year =     [test_this_year] )
        data_obj.add_ys(df_data = df_date_to_datetime(df = meta, cols = ['Date_Harvested']),
                    add_cols = ['Date_Harvested'])
        data_obj.add_xs(df_data = wthr, 
                    drop_cols = ['Year', 'Date', 'DOY'])
        data_obj.isolate_missing_y()
        data_obj.prep_idx_y()
        data_obj.mk_scale_dict(
            scale_cols = ['Date_Harvested']+[e for e in list(wthr) if e not in ['Env', 'Year', 'Date', 'DOY']],
            return_cs_dict = False,
            store_cs_dict = True)
        data_obj.apply_scaling()

        train_y, train_x = data_obj.mk_arrays(
            split = 'train',
            obs_per_Env = 314, 
            return_2d   = False,
            missing_ys  = False)

        test_y, test_x = data_obj.mk_arrays(
            split = 'test',
            obs_per_Env = 314, 
            return_2d   = False,
            missing_ys  = False)


        train_x_2d = wthr_rank_3to2(x_3d = train_x)
        train_y_1d = y_rank_2to1(y_2d = train_y)


        # HPS Study ------------------------------------------------------------------
        cache_save_name = cache_path+trial_name+'_hps.pkl'
        if os.path.exists(cache_save_name):
            study = pkl.load(open(cache_save_name, 'rb'))  
        else:
            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials= n_trials, n_jobs = n_jobs)
            # save    
            pkl.dump(study, open(cache_save_name, 'wb'))    


        # Fit Best HPS ---------------------------------------------------------------   
        cache_save_name = cache_path+trial_name+'_mod.pkl'
        if os.path.exists(cache_save_name):
            rf = pkl.load(open(cache_save_name, 'rb'))  
        else:
            regr = RandomForestRegressor(
                    max_depth = study.best_trial.params['rf_max_depth'], 
                    n_estimators = study.best_trial.params['rf_n_estimators'],
                    min_samples_split = study.best_trial.params['rf_min_samples_split']
                    )
            rf = regr.fit(train_x_2d, train_y_1d)
            # save    
            pkl.dump(rf, open(cache_save_name, 'wb'))   


            # Eval. Best HPS -------------------------------------------------------------    
            for i in range(4):
                # u denotes that the true value is unknown
                temp_label= ['train_y', 'test_y', 'train_u', 'test_u'][i]
                if os.path.exists(cache_path+trial_name+temp_label+".csv"):
                    pass
                else:
                    temp_data = [data_obj.train['y'], data_obj.test['y'], data_obj.train['yna'], data_obj.test['yna']][i]

                    if temp_data.shape[0] != 0:
                        # calculation step below =============================================
                        temp_y, temp_x = data_obj.mk_arrays(                                 #
                            split = temp_label.split('_')[0],                                #
                            obs_per_Env = 314,                                               #
                            return_2d   = False,                                             #
                            missing_ys  = temp_label.split('_')[1] == 'u' )                  #
                                                                                             #
                        temp_y = y_rank_2to1(y_2d = temp_y)                                  #
                        temp_yHat = rf.predict(wthr_rank_3to2(x_3d = temp_x))                #
                        # calculation step above =============================================
                        temp_data['yHat'] = list(temp_yHat)
                        temp_data.to_csv(cache_path+trial_name+'_'+temp_label+".csv", index = False)

                        if temp_label.split('_')[1] != 'u':
                            pd.DataFrame({'MSE':[mean_squared_error(temp_y, temp_yHat)]}).to_csv(cache_path+trial_name+'_'+temp_label+"_mse.csv", index = False)
# -

# write out a log of the enviroments imputed
log_imputed_envs(
    df = meta,
    df_name = 'meta',
    col = 'Date_Harvested'
)   


# +
def temp_fcn(test_this_year):
    demo = df_prep()
    demo.get_train_test_Envs(df = meta, 
                             holdout_years = [], 
                             test_year =     [test_this_year] )
    demo.add_ys(df_data = df_date_to_datetime(df = meta, cols = ['Date_Harvested']),
                add_cols = ['Date_Harvested'])
    demo.add_xs(df_data = wthr, 
                drop_cols = ['Year', 'Date', 'DOY'])
    demo.isolate_missing_y()
    demo.prep_idx_y()
    out = demo.mk_scale_dict(
        scale_cols = ['Date_Harvested'],
        return_cs_dict = True,
        store_cs_dict = False)
    return(out)

# get scaling to use on predictions
years_tested = ['2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021']
# -

yHat_scaling = [pd.DataFrame(temp_fcn(test_this_year)['Date_Harvested'], index = [test_this_year]) for test_this_year in years_tested]
yHat_scaling = pd.concat(yHat_scaling).reset_index().rename(columns = {'index':'CV'})
yHat_scaling.head()

# +
impute_with_mod = 'rf'

file_list = [e for e in os.listdir(cache_path) if re.match('^'+impute_with_mod+'.+.csv$', e) and not re.match('.+mse.csv$', e)]
# allow for all to further model stacking
# for now I'm just going to use one model
#file_list = [e for e in file_list if not  re.match('.+train_y.csv$', e)]

table_list = [pd.read_csv(cache_path+e) for e in file_list]
for i in range(len(file_list)):
    table_list[i]['File'] = file_list[i]
    
yHat = pd.concat(table_list)

yHat[['Model', 'Stage', 'CV', 'Set', 'Value_Known']] = yHat['File'].str.split('_', expand = True)
yHat['CV'] = yHat['CV'].str.strip('test')
yHat.head()

# +
yHat = yHat.merge(yHat_scaling)

yHat['yHat'] = (yHat['yHat']*yHat['std'])+yHat['mean']

yHat = yHat.loc[:, ['Env', 'Year', 'Model', 'CV', 'yHat']]
# TODO if using multiple models and weights, those should be applied here

yHat= yHat.groupby(['Env']).agg(yHat = ('yHat', np.nanmean)).reset_index()
yHat.head()

# +
# Impute missing values
temp = meta

temp = temp.loc[:, ['Year', 'Date_Harvested']].drop_duplicates()
temp['Date_Str'] = temp['Date_Harvested'].astype(str)
temp['DOY'] = [pd.Period(e, freq='D').day_of_year for e in list(temp['Date_Str'])]

temp = meta.merge(temp).merge(yHat)

px.scatter(temp, 'DOY', 'yHat')

# +
mask = (temp.DOY.isna())
temp.loc[mask, 'DOY'] = temp.loc[mask, 'yHat']
temp = temp.drop(columns = ['Date_Str', 'yHat', 'Date_Harvested']).rename(columns = {'DOY': 'Date_Harvested'})

meta = temp
# -

# # Confirm no missing values

print('Cols:', '\t|', 'Min Completion %:')
print('-----', '\t|', '-----------------')
for e in [meta, soil, wthr, cgmv]:
    temp = summarize_col_missing(e)
    print(temp.shape[0], '\t|', min(temp.Pr_Comp))


# # Save out Imputed Data

# ## Make any naming tweaks for readabilty

# +

soil = soil.rename(columns = {
'E Depth': 'E_Depth',
'1:1 Soil pH': 'Soil_pH_1x1',
'WDRF Buffer pH': 'WDRF_Buffer_pH',
'1:1 S Salts mmho/cm': 'S_Salts_mmhopercm_1x1',
'Texture No': 'Texture_No',
'Organic Matter LOI %': 'Organic_Matter_LOI_Pr',
'Nitrate-N ppm N': 'Nitrate_N_ppm_N',
'lbs N/A': 'lbs_NperA',
'Potassium ppm K': 'Potassium_ppm_K',
'Sulfate-S ppm S': 'Sulfate_S_ppm_S',
'Calcium ppm Ca': 'Calcium_ppm_Ca',
'Magnesium ppm Mg': 'Magnesium_ppm_Mg',
'Sodium ppm Na': 'Sodium_ppm_Na',
'CEC/Sum of Cations me/100g': 'CECperSum_of_Cations_meper100g',
'%H Sat': 'PrH_Sat',
'%K Sat': 'PrK_Sat',
'%Ca Sat': 'PrCa_Sat',
'%Mg Sat': 'PrMg_Sat',
'%Na Sat': 'PrNa_Sat',
'Mehlich P-III ppm P': 'Mehlich_P_III_ppm_P',
'% Sand': 'Pr_Sand',
'% Silt': 'Pr_Silt',
'% Clay': 'Pr_Clay'
})
# -

# ## Write out

e = './data/Processed/'
if False:
    phno.to_csv(e+'phno0.csv')
    meta.to_csv(e+'meta0.csv')
    soil.to_csv(e+'soil0.csv')
    wthr.to_csv(e+'wthr0.csv')
    cgmv.to_csv(e+'cgmv0.csv')

# # Format Data for tensors, ERMs

temp = meta.loc[:, ['Env', 'Date_Harvested', 'Date_Planted']].drop_duplicates()
temp['Duration'] = temp['Date_Harvested'] - temp['Date_Planted']
max_duration = np.nanmax(temp['Duration'])
max_duration

#The last harvest day is past the point I clipped the weather data (day 314)
np.nanmax(temp['Date_Harvested'])

# I can't get planting to harvest for all because that would go into the following year
np.nanmax(temp['Date_Planted'])+np.nanmax(temp['Duration'])

# The longest growing period that I can get for all observations is:
314-np.nanmax(temp['Date_Planted'])

# The longest period I could get before is 
np.nanmin(temp['Date_Planted'])

# A full 2 months does not seem useful. If I do 36 then there will be a total of 180 observation
144/4

# +
temp['Date_Planted'] = temp['Date_Planted'].astype(int)
temp['Start_Date'] = temp['Date_Planted'] - 36
temp['End_Date'] = temp['Date_Planted'] + 144

assert 0 == (np.std(temp['End_Date'] - temp['Start_Date']))
# -

temp = temp.reset_index().drop(columns = 'index')
temp['Start_Date'] = temp['Start_Date'].astype(int)
temp['End_Date'] = temp['End_Date'].astype(int)
temp['Date_Planted'] = temp['Date_Planted'].astype(int)
temp['Date_Harvested'] = temp['Date_Harvested'].astype(int)
temp

temp = temp.loc[:, ['Env','Date_Planted', 
             'Start_Date', 'End_Date']].drop_duplicates()

wthr['In_Window'] = False
for i in temp.index:
    Env, Start_Date, End_Date = temp.loc[i, ['Env', 'Start_Date', 'End_Date']]
    mask = (    (wthr.Env == Env
            ) & (wthr.DOY >= Start_Date
            ) & (wthr.DOY < End_Date))
    wthr.loc[mask, 'In_Window'] = True

mask = wthr.In_Window
temp = wthr.loc[mask, ]

# +
min_days = temp.groupby('Env').agg(Min_DOY = ('DOY', np.min)).reset_index()
temp = temp.merge(min_days, how = "outer")

temp['Day'] = 1+(temp['DOY'] - temp['Min_DOY'])

temp = temp.drop(columns = ['Year', 'Date', 'In_Window', 'Min_DOY'])

temp = temp.melt(id_vars = ['Env', 'DOY'])

temp['variable'] = temp['variable'] +'_Day'+ temp['DOY'].astype(str)

temp = temp.drop(columns = ['DOY'])

temp
# -

# This here is causing problems.
# this is going to be SLOW
# temp.loc[(temp.Env.isin(['TXH1_2014', 'NYH3_2022'])), ].pivot(index = ['Env'], columns=['variable']).reset_index()
temp_wide = temp.loc[:, ].pivot(index = ['Env'], columns=['variable']).reset_index()

test = summarize_col_missing(temp_wide)
np.min(test.loc[:, 'Pr_Comp'])
test.sort_values('Pr_Comp')

if False:
    temp_wide.to_csv(e+'wthrWide0.csv')
