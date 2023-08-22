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
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error # if squared=False; RMSE
from sklearn.ensemble import RandomForestRegressor

import plotly.express as px

import os
import optuna
import pickle as pkl

from joblib import Parallel, delayed # Oputna has parallelism built in but for training replicates of the selected model
# I'll run them through Parallel

# +
save_path = './data/Processed/'

phno = pd.read_csv(save_path+"phno3.csv")

YMat = np.load(save_path+'YMat3.npy')
GMat = np.load(save_path+'GMat3.npy')
SMat = np.load(save_path+'SMat3.npy')
WMat = np.load(save_path+'WMat3.npy')
MMat = np.load(save_path+'MMat3.npy')

GMatNames = np.load(save_path+'GMatNames.npy')
SMatNames = np.load(save_path+'SMatNames.npy')
WMatNames = np.load(save_path+'WMatNames.npy')
MMatNames = np.load(save_path+'MMatNames.npy')


# -

# ## Data Prep

# ## SKlearn modeling

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

# ## Tests:

if True == False:
    test_year = 2016

    downsample = True # FIXME
    downsample_train = 1000
    downsample_test  =  100

    # Set up train/test indices --------------------------------------------------
    mask_undefined = (phno.Yield_Mg_ha.isna()) # these can be imputed but not evaluated
    mask_test = ((phno.Year == test_year) & (~mask_undefined))
    mask_train = ((phno.Year != test_year) & (~mask_undefined))
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

    # Center and Scale -----------------------------------------------------------
    YMat = (YMat - YMat_center)/YMat_scale

    SMat = (SMat - SMat_center)/SMat_scale

    MMat = (MMat - MMat_center)/MMat_scale

    # Split and Send to GPU ------------------------------------------------------
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

    # Reshape to rank 1
    train_y = train_y.reshape([train_y.shape[0], 1])
    test_y = test_y.reshape([test_y.shape[0], 1])

if True == False:
    # G
    train_x_2d = train_g
    train_y_1d = y_rank_2to1(y_2d = train_y)
    test_x_2d = test_g
    test_y_1d = y_rank_2to1(y_2d = test_y)


    regr = RandomForestRegressor(max_depth= 16, 
                                 random_state=0,
                                 n_estimators = 20)
    rf = regr.fit(train_x_2d, train_y_1d)


    print([
        mean_squared_error(train_y_1d, rf.predict(train_x_2d), squared=False), 
         mean_squared_error(test_y_1d, rf.predict(test_x_2d), squared=False)
    ])

    # px.bar(pd.DataFrame(dict(cols=GMatNames, imp=rf.feature_importances_)), x = 'cols', y = 'imp')

if True == False:
    # S
    train_x_2d = train_s
    train_y_1d = y_rank_2to1(y_2d = train_y)
    test_x_2d = test_s
    test_y_1d = y_rank_2to1(y_2d = test_y)

    regr = RandomForestRegressor(max_depth= 16, 
                                 random_state=0,
                                 n_estimators = 20)
    rf = regr.fit(train_x_2d, train_y_1d)


    print([
        mean_squared_error(train_y_1d, rf.predict(train_x_2d), squared=False), 
         mean_squared_error(test_y_1d, rf.predict(test_x_2d), squared=False)
    ])

    px.bar(pd.DataFrame(dict(cols=SMatNames, imp=rf.feature_importances_)), x = 'cols', y = 'imp')

if True == False:
    # W
    train_x_2d = wthr_rank_3to2(x_3d = train_w)
    train_y_1d = y_rank_2to1(y_2d = train_y)

    test_x_2d = wthr_rank_3to2(x_3d = test_w)
    test_y_1d = y_rank_2to1(y_2d = test_y)
    
    regr = RandomForestRegressor(max_depth= 16, 
                                 random_state=0,
                                 n_estimators = 20)
    rf = regr.fit(train_x_2d, train_y_1d)
    print([
        mean_squared_error(train_y_1d, rf.predict(train_x_2d), squared=False), 
         mean_squared_error(test_y_1d, rf.predict(test_x_2d), squared=False)
    ])

    px.imshow(
        wthr_features_rank_2to3(train_w, rf.feature_importances_),
        labels=dict(y = 'cols'),
        y = WMatNames
    )


if True == False:
    # M
    train_x_2d = train_m
    train_y_1d = y_rank_2to1(y_2d = train_y)
    test_x_2d = test_m
    test_y_1d = y_rank_2to1(y_2d = test_y)

    regr = RandomForestRegressor(max_depth= 16, 
                                 random_state=0,
                                 n_estimators = 20)
    rf = regr.fit(train_x_2d, train_y_1d)

    print([
        mean_squared_error(train_y_1d, rf.predict(train_x_2d), squared=False), 
         mean_squared_error(test_y_1d, rf.predict(test_x_2d), squared=False)
    ])

    # px.bar(pd.DataFrame(dict(cols=GMatNames, imp=rf.feature_importances_)), x = 'cols', y = 'imp')

if True == False:
    # GSWM
    train_x_2d = np.concatenate([train_g, train_s, wthr_rank_3to2(x_3d = train_w), train_m], axis = 1)
    train_y_1d = y_rank_2to1(y_2d = train_y)
    test_x_2d = np.concatenate([test_g, test_s, wthr_rank_3to2(x_3d = test_w), test_m], axis = 1)
    test_y_1d = y_rank_2to1(y_2d = test_y)

    regr = RandomForestRegressor(max_depth= 16, 
                                 random_state=0,
                                 n_estimators = 20)
    rf = regr.fit(train_x_2d, train_y_1d)


    print([
        mean_squared_error(train_y_1d, rf.predict(train_x_2d), squared=False), 
        mean_squared_error(test_y_1d, rf.predict(test_x_2d), squared=False)
    ])

    # px.bar(pd.DataFrame(dict(cols=GMatNames, imp=rf.feature_importances_)), x = 'cols', y = 'imp')

    feature_imps = rf.feature_importances_

    AllNames = list(GMatNames)+list(SMatNames)+list(WMatNames)+list(MMatNames)
    AllLabels = ['G' for e in list(GMatNames)
              ]+['S' for e in list(SMatNames)
              ]+['W' for e in list(WMatNames)
              ]+['M' for e in list(MMatNames)]

    feature_imp_df = pd.DataFrame(
        zip(AllNames,
            AllLabels,
            feature_imps), 
        columns = ['Feature', 'Group', 'Importance'])

    # how important is each subset?
    feature_imp_df.groupby(['Group']).agg(Total = ('Importance', np.sum))

    # Constrain to larger effects
    feature_imp_df.loc[feature_imp_df.Importance > 0.001, ].groupby(['Group']).agg(Total = ('Importance', np.sum))

# ## Full Data Input

cache_path = './notebook_artifacts/4_modeling_rf/'


def prep_train_test(
    test_this_year = '2014',
    downsample = True, # FIXME
    downsample_train = 1000,
    downsample_test  =  100,
    phno = phno,
    GMat = GMat,
    SMat = SMat,
    WMat = WMat,
    MMat = MMat,
    YMat = YMat):

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

    # Center and Scale -----------------------------------------------------------
    YMat = (YMat - YMat_center)/YMat_scale
    SMat = (SMat - SMat_center)/SMat_scale
    MMat = (MMat - MMat_center)/MMat_scale

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

    # Reshape to rank 1
    train_y = train_y.reshape([train_y.shape[0], 1])
    test_y = test_y.reshape([test_y.shape[0], 1])

    # GSWM
    train_x_2d = np.concatenate([train_g, train_s, wthr_rank_3to2(x_3d = train_w), train_m], axis = 1)
    train_y_1d = y_rank_2to1(y_2d = train_y)
    test_x_2d = np.concatenate([test_g, test_s, wthr_rank_3to2(x_3d = test_w), test_m], axis = 1)
    test_y_1d = y_rank_2to1(y_2d = test_y)
    
    
    full_x_2d = np.concatenate([GMat, SMat, wthr_rank_3to2(x_3d = WMat), MMat], axis = 1)
    return(train_x_2d, train_y_1d, test_x_2d, test_y_1d, full_x_2d, YMat_center, YMat_scale, YMat)


# +
# Ran from 17:15:40 2023-01-05 -> 9:15:40 2023-01-06 completing only 2014, (2015 still running single thread)
# To reduce the run time, I'm shrinking the search space based on what was effective for 2014 and 2015
# 2014 == *
# 2015 == v (best so far trial 52)
#         'rf_max_depth',    2-10v----65*----------------------200
#      'rf_n_estimators',   20-30v-39*-------------------------500
# 'rf_min_samples_split', 0.05-.5*v---------------------------0.95

# Setup ----------------------------------------------------------------------
trial_name = 'rf'
# takes about 3 minutes to fit one full model
n_trials= 120 
n_jobs = 30
test_this_year = '2014'

downsample = False# FIXME
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
    
    rf = regr.fit(train_x_2d, train_y_1d)
    return (mean_squared_error(train_y_1d, rf.predict(train_x_2d), squared=False))


if True == False:
    reset_trial_name = trial_name
    for test_this_year in [#'2014', '2015', '2016', '2017', '2018', '2019', '2020', 
                           '2021']:
        print("""
    ------------------------------------------
    ------------------"""+test_this_year+"""------------------
        """)    

        trial_name = reset_trial_name
        trial_name = trial_name+test_this_year
        print(test_this_year)
        # Data Prep. -----------------------------------------------------------------
        # Set up train/test indices --------------------------------------------------
        train_x_2d, train_y_1d, test_x_2d, test_y_1d, full_x_2d, YMat_center, YMat_scale, YMat = prep_train_test(
            test_this_year = test_this_year,
            downsample = downsample, 
            downsample_train = downsample_train,
            downsample_test  =  downsample_test,
            phno = phno,
            GMat = GMat,
            SMat = SMat,
            WMat = WMat,
            MMat = MMat,
            YMat = YMat)

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
                rf = regr.fit(train_x_2d, train_y_1d)
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
    #         out.to_csv('./notebook_artifacts/4_modeling_rf/'+trial_name+'_'+str(Rep)+'rfYHats.csv')
            out.to_csv('./data/Shared_Model_Output/'+trial_name+'_'+str(Rep)+'rfYHats.csv')

        # use joblib to get replicate models all at once
        Parallel(n_jobs=10)(delayed(fit_single_rep)(Rep = i) for i in range(10))

# +
# [I 2023-01-06 08:56:55,012] Trial 27 finished with value: 0.7037294697990369 and parameters: {'rf_max_depth': 33, 'rf_n_estimators': 266, 'rf_min_samples_split': 0.05867248086866843}. Best is trial 52 with value: 0.6961002467654831.
