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
import urllib.request, json 

import tqdm
import time

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import optuna
import pickle as pkl

from joblib import Parallel, delayed

with open('./notebook_artifacts/usda_nass_api_key.txt') as f:
    api_key = f.readlines()
    api_key = api_key[0]
    
cache_path = './notebook_artifacts/9_historical/'    

save_path = './data/Processed/'
meta = pd.read_csv(save_path+"meta0.csv")
phno = pd.read_csv(save_path+"phno3.csv")
# -

# # Add State and County data into 2022

# +
# drop sites in canada
canada_list = [
    'ONH2_2019', 'ONH2_2017', 'ONH2_2014', 'ONH2_2016', 'ONH2_2015', 
    'ONH1_2017', 'ONH1_2015', 'ONH1_2014', 'ONH1_2016']
mask = [False if e in canada_list else True for e in phno.Env]
phno = phno.loc[mask, ].reset_index().drop(columns = ['index'])


mask = [False if e in canada_list else True for e in meta.Env]
meta = meta.loc[mask, ].reset_index().drop(columns = ['index'])

# -

# phno
latlons = meta.loc[(meta.Year == 2022), ['Env', 'Latitude_of_Field', 'Longitude_of_Field']].drop_duplicates().reset_index().drop(columns = 'index')
latlons2 =meta.loc[(meta.Year != 2022), ['Env', 'Latitude_of_Field', 'Longitude_of_Field']].drop_duplicates().reset_index().drop(columns = 'index')

# +
# initially only looked at counties in the test set
latlons['county'] = ''
latlons['zip'] = ''
latlons['states'] = ''

zipcodes = [
    '52249',  '50014',  '68507', '53593', '53593', '47906', '19947', '52621', 
    '51443', '61820', '13026', '13026', '48864', '56093', '30677', '65203',  
    '45368', '69101', '69101', '54930', '29670', '77879', '77879', '77879', 
    '31793', '27849']
counties = [
    'Benton', 'Story', 'Lancaster','Dane','Dane','Tippecanoe','Sussex', 
    'Washington', 'Carroll', # originally typed as carrol 
    'Champaign', 'Cayuga', 'Cayuga', 'Ingham', 
    'Waseca', 'Oconee', 'Boone', 'Clark', 'Lincoln', 'Lincoln', 'Waushara', 
    'Anderson', 'Burleson', 'Burleson', 'Burleson', 'Tift', 'Bertie']
states =   [
    'IA', 'IA', 'NE', 'WI', 'WI', 'IN', 'DE', 'IA', 'IA', 'IL', 'NY', 'NY', 
    'MI', 'MN', 'GA', 'MO', 'OH', 'NE', 'NE', 'WI', 'SC', 'TX', 'TX', 'TX', 
    'GA', 'NC']

for i in range(len(zipcodes)):
    latlons.loc[i, 'zip'] = zipcodes[i]    
for i in range(len(counties)):
    latlons.loc[i, 'county'] = counties[i]
for i in range(len(states)):
    latlons.loc[i, 'states'] = states[i]

# latlons#.loc[(latlons.zip == ''), ]
# -

latlons.loc[latlons.states == 'IA']

# +
# what about other counties?
threshold = 0.05 # for euclid dist between existing/non latlon
threshold = 0.1 # for euclid dist between existing/non latlon
tmp = latlons.merge(latlons2, how = 'outer')


## Add labels manually
# https://gps-coordinates.org/what-county-am-i-in.php
add_data =  [
    #Lat         Lon       county  state
    [34.727251, -90.760326, 'Lee', 'AR'],
    [35.296577, -77.566463 , 'Lenoir', 'NC'],
    [35.483467,-77.569042, 'Greene', 'NC'],
    [35.666783, -78.492372, 'Johnston', 'NC'],
    [35.838614, -90.665302, 'Craighead', 'AR'],
    [37.810107, -100.775188, 'Finney', 'KS'],
    [38.892640, -92.205753, 'Boone', 'MO'],
    [38.913500, -92.281389, 'Boone', 'MO'],
    [39.144807, -96.628982, 'Riley', 'KS'],
    [39.216418,-96.604822, 'Riley', 'KS'],
    [40.648500,-104.999782, 'Larimer', 'CO'],
    [41.158245,-101.987296, 'Keith', 'NE'],
    [41.162000, -96.409000, 'Saunders', 'NE'],
    [41.290979,-96.416643, 'Saunders', 'NE'],
    [41.380423, -96.559884, 'Saunders', 'NE'],
    [42.031938, -93.793332, 'Boone', 'IA'],
    [42.412067,-84.295262, 'Ingham', 'MI'],
#     [42.452213	-81.881844, '', ''], # Canada
#     [43.495460,-80.425969, '', ''],
#     [, '', ''],
#     [, '', ''],
#     [, '', ''],
#     [, '', ''],
    [44.208745, -102.928745, 'Pennington', 'SD']
]
search_tol = 0.01

for i in range(len(add_data)):
    mask = ( (tmp.county.isna()
        ) & ((tmp.Latitude_of_Field  - add_data[i][0]) < search_tol
        ) & ((tmp.Longitude_of_Field - add_data[i][1]) < search_tol))

    tmp.loc[mask, 'county'] = add_data[i][2]
    tmp.loc[mask, 'states'] = add_data[i][3]
    
## Add labels by distance
tmp_has_county = tmp.loc[(tmp.county.notna()), :]
tmp = tmp.loc[(tmp.county.isna()), :]

tmp = tmp.loc[:, ['Env', 'Latitude_of_Field', 'Longitude_of_Field', 'county', 'states']
       ].drop_duplicates(
       ).sort_values(['Latitude_of_Field', 'Longitude_of_Field']).reset_index().drop(columns = ['index'])

for i in tmp.index:
    tmp_has_county['Latitude_Diff'] = np.nan
    tmp_has_county['Longitude_Diff'] = np.nan
    tmp_has_county['Euclid_Diff'] = np.nan

    tmp_has_county['Latitude_Diff']  = (tmp_has_county.loc[:, 'Latitude_of_Field']  - tmp.loc[i, 'Latitude_of_Field'])
    tmp_has_county['Longitude_Diff'] = (tmp_has_county.loc[:, 'Longitude_of_Field'] - tmp.loc[i, 'Longitude_of_Field'])
    tmp_has_county['Euclid_Diff']    = np.sqrt(tmp_has_county['Latitude_Diff']**2 + tmp_has_county['Longitude_Diff']**2)

    tmp_has_county.sort_values(['Euclid_Diff'])

    min_diff = min(tmp_has_county['Euclid_Diff'])
    mask = (tmp_has_county['Euclid_Diff'] == min_diff)

    if min_diff > threshold:
        pass
    else:
        tmp.loc[i, 'county'] = list(tmp_has_county.loc[mask, 'county'])[0]
        tmp.loc[i, 'states'] = list(tmp_has_county.loc[mask, 'states'])[0]
# -

print(tmp.loc[tmp.county.isna(), ].shape[0])
tmp.loc[tmp.county.isna(), ]

tmp = tmp.loc[tmp.county.notna(), ].copy() # gets rid of ONH\d values

# +
cols = ['Env', 'Latitude_of_Field', 'Longitude_of_Field', 'county', 'states']

latlons = tmp.loc[:, cols
            ].merge(tmp_has_county.loc[:, cols], how = 'outer'
            ).drop(columns = ['Latitude_of_Field', 'Longitude_of_Field']
            ).drop_duplicates()


latlons.loc[:, 'county'] = [e.upper() for e in latlons['county']]
# -

dat = phno.merge(latlons, how = 'outer')
dat = dat.rename(columns = {
    'county':'County', 
    'states':'State'})
dat

# # Process NASS data

# +
# Building up a dataset from historical entries
# Commodity: CORN
# Category: YIELD
# Geographic Level: County
# State: #select a set of states that is small enough to not hit the 50000 entry download limit
# e.g. './notebook_artifacts/9_historical/5AD68EDD-41CB-308E-B5F0-530F45F54F3C.csv'

# There's no 2022 data included in the downloaded data so I'll use the state level too
nass_state_csv = 'E33B3659-1330-3D9B-BFC3-DF934E2AEE99.csv'

nass_csvs = [e for e in os.listdir(cache_path) if re.match('[A-Z0-9]{8}\-[A-Z0-9]{4}\-[A-Z0-9]{4}\-[A-Z0-9]{4}\-[A-Z0-9]{12}\.csv', e)]
nass_csvs = [e for e in nass_csvs if e != nass_state_csv]

nass_df = pd.concat([pd.read_csv(cache_path+nass_csv) for nass_csv in nass_csvs])

# +
nass_df = nass_df.loc[:, [
#     'Program',
    'Year',
#     'Period',
#     'Week Ending',
#     'Geo Level',
    'State',
#     'State ANSI',
    'Ag District',
#     'Ag District Code',
    'County',
#     'County ANSI',
#     'Zip Code',
#     'Region',
#     'watershed_code',
#     'Watershed',
#     'Commodity',
    'Data Item',
#     'Domain',
#     'Domain Category',
    'Value',
#     'CV (%)'
]]

[[len(set(nass_df[e])), e] for e in list(nass_df)]
# -

data_dict = {
                "CORN, GRAIN, IRRIGATED - YIELD, MEASURED IN BU / ACRE": "GRN_IRR_BUpACRE",
               "CORN, GRAIN - YIELD, MEASURED IN BU / NET PLANTED ACRE": "GRN_BUpNETPLANTEDACRE",
"CORN, GRAIN, NON-IRRIGATED - YIELD, MEASURED IN BU / NET PLANTED ACRE": "GRN_NON_IRR_BUpNETPLANTEDACRE",
             "CORN, SILAGE, IRRIGATED - YIELD, MEASURED IN TONS / ACRE": "SLG_IRR_TONSpACRE",
         "CORN, SILAGE, NON-IRRIGATED - YIELD, MEASURED IN TONS / ACRE": "SLG_NON_IRR_TONSpACRE",
            "CORN, GRAIN, NON-IRRIGATED - YIELD, MEASURED IN BU / ACRE": "GRN_NON_IRR_BUpACRE",
    "CORN, GRAIN, IRRIGATED - YIELD, MEASURED IN BU / NET PLANTED ACRE": "GRN_IRR_BUpNETPLANTEDACRE",
                        "CORN, SILAGE - YIELD, MEASURED IN TONS / ACRE": "SLG_TONSpACRE",
                           "CORN, GRAIN - YIELD, MEASURED IN BU / ACRE": "GRN_BUpACRE"
}

nass_df = nass_df.merge(pd.DataFrame(data_dict, index = [0]).T.reset_index(
                ).rename(columns = {'index':'Data Item', 0:'Key'})
                ).drop(columns = ['Data Item'])
nass_df = nass_df.rename(columns = {'Ag District': 'AgDistrict'})

nass_df

nass_df_wide = nass_df.pivot(columns='Key', 
                             values='Value', 
                             index=['Year', 'State', 'AgDistrict', 'County']
                            ).reset_index()
nass_df_wide

# drop cols with low fill rate
nass_df_wide = nass_df_wide.drop(columns = [
    'GRN_BUpNETPLANTEDACRE',
    'GRN_IRR_BUpNETPLANTEDACRE',
    'GRN_NON_IRR_BUpNETPLANTEDACRE',
    'SLG_IRR_TONSpACRE',
    'SLG_NON_IRR_TONSpACRE',
])

# +
# nass_df_wide.loc[nass_df_wide.Year == 2022, ]

# add a placeholder for 2022 data

tmp = nass_df_wide.loc[:, ['State', 'AgDistrict', 'County']].drop_duplicates().reset_index().drop(columns = ['index'])
tmp['Year'] = 2022

nass_df_wide = nass_df_wide.merge(tmp, how = 'outer')

# -

nass_df_wide.info()

# +
nass_state_df = pd.read_csv(cache_path+nass_state_csv)

nass_state_df = nass_state_df.loc[:, 
[
#'Program',
 'Year',
#  'Period',
#  'Week Ending',
#  'Geo Level',
 'State',
#  'State ANSI',
#  'Ag District',
#  'Ag District Code',
#  'County',
#  'County ANSI',
#  'Zip Code',
#  'Region',
#  'watershed_code',
#  'Watershed',
#  'Commodity',
 'Data Item',
#  'Domain',
#  'Domain Category', #  going to ignore this, 'IRRIGATION METHOD, PRIMARY: (GRAVITY)',
                                            #  'IRRIGATION METHOD, PRIMARY: (PRESSURE)',
                                            #  'IRRIGATION STATUS: (ANY ON OPERATION)',
                                            #  'NOT SPECIFIED'
 'Value',
#  'CV (%)'
]].drop_duplicates()

# .pivot(columns='Key', 
#                              values='Value', 
#                              index=['Year', 'State', 'AgDistrict', 'County']
#                             ).reset_index(
# -

 nass_state_df = nass_state_df.loc[(nass_state_df.Value != ' (D)'), ].copy()

nass_state_df.loc[:, ['Value']] = nass_state_df.loc[:, ['Value']].astype(float)

nass_state_df = nass_state_df.merge(pd.DataFrame(data_dict, index = [0]).T.reset_index(
                ).rename(columns = {'index':'Data Item', 0:'Key'})
                ).drop(columns = ['Data Item']
                ).groupby(['Year', 'State', 'Key']
                ).agg(Value = ('Value', np.nanmean)
                ).reset_index()

# +
# nass_df_wide = nass_df
nass_state_df = nass_state_df.pivot(
        columns='Key', 
        values='Value', 
        index=['Year', 'State']
    ).reset_index().loc[:, [
        'Year', 'State', 'GRN_BUpACRE', 'SLG_TONSpACRE']
    ].rename(columns = {
    'GRN_BUpACRE': 'STATE_GRN_BUpACRE', 
    'SLG_TONSpACRE': 'STATE_SLG_TONSpACRE'})

nass_state_df

# +
# impute missing values
nass_df_big = nass_df_wide.merge(nass_state_df).drop(columns = [
    'GRN_IRR_BUpACRE', 'GRN_NON_IRR_BUpACRE', 'SLG_TONSpACRE']) # drop low completion (~1/10) cols

nass_df_big = nass_df_big.loc[~(nass_df_big.AgDistrict.isna()), ] # drop "OTHER COUNTIES"

# +
# https://www.faa.gov/air_traffic/publications/atpubs/cnt_html/appendix_a.html

state_code_lookup = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Guam': 'GU',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Puerto Rico': 'PR',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY'}
# -

state_code_lookup = pd.DataFrame(state_code_lookup, index = [0]).T.reset_index().rename(columns = {'index':'State', 0:'StateAbbr'})
state_code_lookup['State'] = state_code_lookup.State.str.upper()
# state_code_lookup.merge(nass_df_big, how = 'outer')

nass_df_big = nass_df_big.merge(state_code_lookup).drop(columns = ['State']).rename(columns = {'StateAbbr':'State'})

nass_df_big

# ## Test to see if there are missing counties

# +
# check for counties in dat not found in nass_df_big (way 1)
# -

tmp1 = nass_df_big.loc[:, ['State', 'County']].drop_duplicates()
tmp1['nass'] = True

tmp2 = dat.loc[:, ['Env', 'County', 'State']].drop_duplicates()
tmp2['dat'] = True

tmp = tmp1.merge(tmp2, how = 'outer')
tmp

tmp.loc[((tmp.dat == True) & (tmp.nass == False)), ]

# +
# check for counties in dat not found in nass_df_big (way 2)

for ith_state in list(set(dat.State)):
    counties_found = list(set(nass_df_big.loc[(nass_df_big.State == ith_state), 'County']))
    missings = [e for e in list(set(dat.loc[(dat.State == ith_state), 'County'])) if e not in counties_found]
    if missings != []:
        print(ith_state)
        print(counties_found)
        print(missings)
        print('\n')

# +

# filter nass_df_big down to counties in the dat dataset
nass_df_big = dat.loc[:, ['State', 'County']].drop_duplicates().merge(nass_df_big)
# -

# ## Impute mising values

from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OneHotEncoder


# I'd like to use a nested mixed effects model here but geting that running with pymc3 or statsmodels is going to be challenging in the time frame
def iter_imp_nass_df(nass_df_big):
    cat_cols = ['State', 'AgDistrict', 'County']
    float_cols = [e for e in list(nass_df_big) if e not in cat_cols]

    nass_df_big_cats = nass_df_big.loc[:, cat_cols]

    enc = OneHotEncoder()
    enc.fit(nass_df_big_cats)
    nass_onehot = enc.transform(nass_df_big_cats).toarray()
    
    nass_np_cat = np.concatenate([
        nass_onehot,
        np.asanyarray(nass_df_big.drop(columns = cat_cols))
        ], axis = 1)

    imp_mean = IterativeImputer(random_state=0)
    imp_mean.fit(nass_np_cat)

    nass_np_cat = imp_mean.transform(nass_np_cat)

    nass_imp = pd.concat([pd.DataFrame(
        enc.inverse_transform(nass_np_cat[:, 0:nass_onehot.shape[1]]),
        columns = cat_cols
    ), pd.DataFrame(
        nass_np_cat[:, nass_onehot.shape[1]:],
        columns = float_cols
    )
    ], axis = 1)

    return(nass_imp)


# +
# Impute using each state's data. 
# this is a compromise to get some nesting without spending the time to get 
# a mixed model trained.

nass_df_imp = pd.concat([
    iter_imp_nass_df(nass_df_big = nass_df_big.loc[(nass_df_big.State == e), :])
    for e in 
    tqdm.tqdm(list(set(nass_df_big.State)))
])
# -

nass_df_imp

# # Merge imputed yield data with G2F data

# ##  Merge

dat.head(3)

nass_df_imp.head(3)

tmp = dat.merge(
    # Use the state_code_lookup to replace fully written state names with abbrs
    nass_df_imp, 
    how = 'left')

tmp.loc[tmp.Year == 2022, ].info()

# Some entries lack data. This does not apply to any of the 2022 data so I'll 
# drop the sites lacking historical data.
mask = (tmp.GRN_BUpACRE.notna())
tmp = tmp.loc[mask, ]

# now the only missing values in dat are Yield_Mg_ha
dat = tmp

# ## Prepare matrices

dat = dat.reset_index().drop(columns = 'index')


def prep_train_test_hist(
test_this_year = '2014',
dat = dat
                        ):
    mask_undefined = (dat.Yield_Mg_ha.isna()) # these can be imputed but not evaluated
    mask_test = ((dat.Year == int(test_this_year)) & (~mask_undefined))
    mask_train = ((dat.Year != int(test_this_year)) & (~mask_undefined))
    test_idx = dat.loc[mask_test, ].index
    train_idx = dat.loc[mask_train, ].index

    ## Encode Hybrid Info ========================================================
    res = []
    for e in list(set(dat['Hybrid'])):
        res += e.split('/')
    res = list(set(res))


    hybrid_onehot = dat.loc[:, ['Hybrid']]#.drop_duplicates(
                                         #).reset_index().drop(columns = ['index'])

    hybrid_onehot = hybrid_onehot.Hybrid.str.split('/', expand=True
                                           ).rename(columns = {0:'p0', 1:'p1'})

    enc = OneHotEncoder()
    enc.fit(np.array(res).reshape([len(res), 1]))

    p0 = enc.transform(hybrid_onehot.loc[:, ['p0']]).toarray()
    p1 = enc.transform(hybrid_onehot.loc[:, ['p1']]).toarray()

    hybrid_mat = p0+p1

    ## Encode Location Info ======================================================
    position_onehot = dat.loc[:, ['Env', 'County', 'State']]

    enc = OneHotEncoder()
    enc.fit(position_onehot)
    position_onehot = enc.transform(position_onehot).toarray()

    ## Turn Yields into array ====================================================
    county_state_mat = np.array(dat.loc[:, ['GRN_BUpACRE', 'STATE_GRN_BUpACRE', 
                                      'STATE_SLG_TONSpACRE']])

    ## Merge into a single array =================================================
    xs_mat = np.concatenate([county_state_mat, hybrid_mat, position_onehot], 
                            axis = 1)

    ## Y =========================================================================
    ys_mat = np.array(dat.loc[:, 'Yield_Mg_ha'])
#     ys_mat = ys_mat.reshape(ys_mat.shape[0], 1)

    train_x = xs_mat[train_idx]
    test_x  = xs_mat[test_idx]

    train_y = ys_mat[train_idx]
    test_y  = ys_mat[test_idx]

    # Reshape to rank 1
    # train_y = train_y.reshape([train_y.shape[0], 1])
    # test_y = test_y.reshape([test_y.shape[0], 1])
    return(train_x, train_y, test_x, test_y, xs_mat, ys_mat)

# +
# Setup ----------------------------------------------------------------------
trial_name = 'hist'

n_trials= 40
n_jobs = 30


def objective(trial): 
    rf_max_depth = trial.suggest_int('rf_max_depth', 2, 100, log=True)
    rf_n_estimators = trial.suggest_int('rf_n_estimators', 20, 100, log=True)
    rf_min_samples_split = trial.suggest_float('rf_min_samples_split', 0.005, 0.5, log=True)
    
    regr = RandomForestRegressor(
        max_depth = rf_max_depth, 
        n_estimators = rf_n_estimators,
        min_samples_split = rf_min_samples_split
        )
    
    rf = regr.fit(train_x, train_y)
    return (mean_squared_error(train_y, rf.predict(train_x), squared=False))


if True == False:
    reset_trial_name = trial_name

    for test_this_year in ['2021', '2020', '2019', '2018', '2017', '2016', '2015', '2014']:
        print("""
    ------------------------------------------
    ------------------"""+test_this_year+"""------------------
        """)    

        trial_name = reset_trial_name
        trial_name = trial_name+test_this_year
        print(test_this_year)
        # Data Prep. -----------------------------------------------------------------
        # Set up train/test indices --------------------------------------------------
        train_x, train_y, test_x, test_y, xs_mat, ys_mat = prep_train_test_hist(
            test_this_year = test_this_year,
            dat = dat)

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
                rf = regr.fit(train_x, train_y)
                # save    
                pkl.dump(rf, open(cache_save_name, 'wb'))   

            # Record Predictions -----------------------------------------------------
            out = dat.copy()
            out['YHat'] = rf.predict(xs_mat)
            out['YMat'] = ys_mat
            out['Y_center'] = None
            out['Y_scale'] = None
            out['Class'] = trial_name
            out['CV'] = test_this_year
            out['Rep'] = Rep
            out.to_csv('./notebook_artifacts/9_historical/'+trial_name+'_'+str(Rep)+'YHats.csv')
    #         out.to_csv('./data/Shared_Model_Output/'+trial_name+'_'+str(Rep)+'rfYHats.csv')

        # use joblib to get replicate models all at once
        Parallel(n_jobs=10)(delayed(fit_single_rep)(Rep = i) for i in range(10)) #FIXME
# -

if True == False:
    import os, re
    import shutil

    start_path = cache_path
    end_path   = './data/Shared_Model_Output/'

    yHat_files = [e for e in os.listdir(start_path) if re.match('^hist\d\d\d\d_\dYHats.csv$', e)]

    for yHat_file in yHat_files:
        shutil.move(
            start_path+yHat_file,
            end_path+yHat_file
        )

if True == False:
    dat.to_csv('./data/Shared_Model_Output/hist9.csv')
