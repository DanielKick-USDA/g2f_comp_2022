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
# from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

import plotly.express as px

import os
import optuna
import pickle as pkl

from joblib import Parallel, delayed # Oputna has parallelism built in but for training replicates of the selected model
# I'll run them through Parallel

import tqdm

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

cache_path = './notebook_artifacts/4_modeling_xgb/'


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


if True == False: # this took a ton of time to run a single year _even_ after 
                  # constraining the search space. See below
    # Setup ----------------------------------------------------------------------
    trial_name = 'xgb'


    n_trials= 60 #120  #FIXME
    n_jobs = 1
    n_jobs_xgb = 20
    n_jobs_retrain = 1
    # test_this_year = '2014'

    downsample = False# FIXME
    downsample_train = 1000
    downsample_test  =  100


    def objective(trial):
    #     xgb_max_depth = trial.suggest_int('xgb_max_depth', 2, 200, log=True)
        xgb_max_depth = trial.suggest_int('xgb_max_depth', 2, 100, log=True)
    #     xgb_n_estimators = trial.suggest_int('xgb_n_estimators', 2, 200, log=True)    
        xgb_n_estimators = trial.suggest_int('xgb_n_estimators', 20, 200, log=True)
        xgb_learning_rate = trial.suggest_float('xgb_learning_rate', 0.0001, 0.3, log=True)

        regr = XGBRegressor(
            max_depth = xgb_max_depth, 
            n_estimators = xgb_n_estimators,
            learning_rate = xgb_learning_rate,
            objective='reg:squarederror',
            n_jobs= n_jobs_xgb
            )

        xgb = regr.fit(train_x_2d, train_y_1d)
        return (mean_squared_error(train_y_1d, xgb.predict(train_x_2d), squared=False))



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

            # Record Predictions -----------------------------------------------------
            out = phno.copy()
            out['YHat'] = xgb.predict(full_x_2d)
            out['YMat'] = YMat
            out['Y_center'] = YMat_center
            out['Y_scale'] = YMat_scale
            out['Class'] = trial_name
            out['CV'] = test_this_year
            out['Rep'] = Rep
            out.to_csv('./notebook_artifacts/4_modeling_xgb/'+trial_name+'_'+str(Rep)+'xgbYHats.csv')
    #         out.to_csv('./data/Shared_Model_Output/'+trial_name+'_'+str(Rep)+'xgbYHats.csv')

        # use joblib to get replicate models all at once
        Parallel(n_jobs=n_jobs_retrain)(delayed(fit_single_rep)(Rep = i) for i in range(10))

# +
# 2021

# [I 2023-01-09 13:55:27,652] A new study created in memory with name: no-name-cee68691-bca7-4610-88b4-7ecc0c7f80ef
# [I 2023-01-09 14:40:00,425] Trial 0 finished with value: 1.098403865386626 and parameters: {'xgb_max_depth': 79, 'xgb_n_estimators': 176, 'xgb_learning_rate': 0.00012889096406452452}. Best is trial 0 with value: 1.098403865386626.
# [I 2023-01-09 14:52:46,994] Trial 1 finished with value: 1.0971413011381197 and parameters: {'xgb_max_depth': 42, 'xgb_n_estimators': 90, 'xgb_learning_rate': 0.0002787378360314917}. Best is trial 1 with value: 1.0971413011381197.
# [I 2023-01-09 14:54:46,188] Trial 2 finished with value: 1.1152997921706163 and parameters: {'xgb_max_depth': 25, 'xgb_n_estimators': 22, 'xgb_learning_rate': 0.0001527169263612772}. Best is trial 1 with value: 1.0971413011381197.
# [I 2023-01-09 14:58:15,357] Trial 3 finished with value: 0.4228922897675502 and parameters: {'xgb_max_depth': 50, 'xgb_n_estimators': 20, 'xgb_learning_rate': 0.29593975644638393}. Best is trial 3 with value: 0.4228922897675502.

# [I 2023-01-09 15:37:18,370] Trial 4 finished with value: 0.3315474268629703 and parameters: {'xgb_max_depth': 94, 'xgb_n_estimators': 129, 'xgb_learning_rate': 0.17811243278613442}. Best is trial 4 with value: 0.3315474268629703.

# [I 2023-01-09 15:37:53,993] Trial 5 finished with value: 0.6623451441788236 and parameters: {'xgb_max_depth': 4, 'xgb_n_estimators': 27, 'xgb_learning_rate': 0.11593265756299452}. Best is trial 4 with value: 0.3315474268629703.
# [I 2023-01-09 15:41:16,974] Trial 6 finished with value: 0.7960393408983473 and parameters: {'xgb_max_depth': 35, 'xgb_n_estimators': 27, 'xgb_learning_rate': 0.019226050995519616}. Best is trial 4 with value: 0.3315474268629703.
# [I 2023-01-09 15:43:02,345] Trial 7 finished with value: 1.0056153287518295 and parameters: {'xgb_max_depth': 13, 'xgb_n_estimators': 35, 'xgb_learning_rate': 0.004577347227714729}. Best is trial 4 with value: 0.3315474268629703.
# [I 2023-01-09 15:45:29,157] Trial 8 finished with value: 1.1103963558642957 and parameters: {'xgb_max_depth': 12, 'xgb_n_estimators': 54, 'xgb_learning_rate': 0.00018464496500957786}. Best is trial 4 with value: 0.3315474268629703.
# [I 2023-01-09 16:37:20,594] Trial 9 finished with value: 1.050596541583475 and parameters: {'xgb_max_depth': 78, 'xgb_n_estimators': 199, 'xgb_learning_rate': 0.0004046699864638791}. Best is trial 4 with value: 0.3315474268629703.
# [I 2023-01-09 16:38:42,254] Trial 10 finished with value: 0.804893322943509 and parameters: {'xgb_max_depth': 2, 'xgb_n_estimators': 112, 'xgb_learning_rate': 0.013670216943992671}. Best is trial 4 with value: 0.3315474268629703.
# [I 2023-01-09 16:52:41,816] Trial 11 finished with value: 0.3569047721266109 and parameters: {'xgb_max_depth': 77, 'xgb_n_estimators': 55, 'xgb_learning_rate': 0.25853768904115954}. Best is trial 4 with value: 0.3315474268629703.
# [I 2023-01-09 17:11:31,744] Trial 12 finished with value: 0.3602804956788001 and parameters: {'xgb_max_depth': 96, 'xgb_n_estimators': 58, 'xgb_learning_rate': 0.08154880854940047}. Best is trial 4 with value: 0.3315474268629703.
# [I 2023-01-09 17:17:07,182] Trial 13 finished with value: 0.5039438911162765 and parameters: {'xgb_max_depth': 18, 'xgb_n_estimators': 88, 'xgb_learning_rate': 0.24297597782700675}. Best is trial 4 with value: 0.3315474268629703.
# [I 2023-01-09 17:18:17,574] Trial 14 finished with value: 0.6576554736671044 and parameters: {'xgb_max_depth': 7, 'xgb_n_estimators': 40, 'xgb_learning_rate': 0.044471059342886775}. Best is trial 4 with value: 0.3315474268629703.
# [I 2023-01-09 17:42:02,356] Trial 15 finished with value: 0.8557672146470044 and parameters: {'xgb_max_depth': 51, 'xgb_n_estimators': 136, 'xgb_learning_rate': 0.0028016435117183126}. Best is trial 4 with value: 0.3315474268629703.
# [I 2023-01-09 18:07:57,688] Trial 16 finished with value: 0.4068148902296806 and parameters: {'xgb_max_depth': 99, 'xgb_n_estimators': 77, 'xgb_learning_rate': 0.03226689731886975}. Best is trial 4 with value: 0.3315474268629703.
# [I 2023-01-09 18:12:31,254] Trial 17 finished with value: 1.072835071487901 and parameters: {'xgb_max_depth': 27, 'xgb_n_estimators': 48, 'xgb_learning_rate': 0.001187951426594088}. Best is trial 4 with value: 0.3315474268629703.
# [I 2023-01-09 18:27:38,114] Trial 18 finished with value: 0.38530341938203466 and parameters: {'xgb_max_depth': 66, 'xgb_n_estimators': 69, 'xgb_learning_rate': 0.10960867765858516}. Best is trial 4 with value: 0.3315474268629703.
# [I 2023-01-09 18:30:47,455] Trial 19 finished with value: 0.6966345869508496 and parameters: {'xgb_max_depth': 7, 'xgb_n_estimators': 114, 'xgb_learning_rate': 0.010927773046523889}. Best is trial 4 with value: 0.3315474268629703.
# [I 2023-01-09 18:44:49,900] Trial 20 finished with value: 0.4777381150802678 and parameters: {'xgb_max_depth': 29, 'xgb_n_estimators': 143, 'xgb_learning_rate': 0.059966915736209506}. Best is trial 4 with value: 0.3315474268629703.
# [I 2023-01-09 19:02:39,762] Trial 21 finished with value: 0.3520111923254733 and parameters: {'xgb_max_depth': 94, 'xgb_n_estimators': 57, 'xgb_learning_rate': 0.12474208650378521}. Best is trial 4 with value: 0.3315474268629703.
# [I 2023-01-09 19:10:34,250] Trial 22 finished with value: 0.4007334876581492 and parameters: {'xgb_max_depth': 57, 'xgb_n_estimators': 42, 'xgb_learning_rate': 0.1760567166746985}. Best is trial 4 with value: 0.3315474268629703.
# [I 2023-01-09 19:24:51,437] Trial 23 finished with value: 0.38346213617609737 and parameters: {'xgb_max_depth': 65, 'xgb_n_estimators': 66, 'xgb_learning_rate': 0.13408279706753004}. Best is trial 4 with value: 0.3315474268629703.
# [I 2023-01-09 19:41:56,498] Trial 24 finished with value: 0.4887615224688054 and parameters: {'xgb_max_depth': 97, 'xgb_n_estimators': 52, 'xgb_learning_rate': 0.029539144094783}. Best is trial 4 with value: 0.3315474268629703.
# [I 2023-01-09 19:53:09,224] Trial 25 finished with value: 0.447867828149032 and parameters: {'xgb_max_depth': 40, 'xgb_n_estimators': 84, 'xgb_learning_rate': 0.07368002232526397}. Best is trial 4 with value: 0.3315474268629703.
# [I 2023-01-09 19:59:02,760] Trial 26 finished with value: 0.502030642309641 and parameters: {'xgb_max_depth': 17, 'xgb_n_estimators': 101, 'xgb_learning_rate': 0.26360187167175286}. Best is trial 4 with value: 0.3315474268629703.
# [I 2023-01-09 20:06:50,102] Trial 27 finished with value: 0.3907415512869049 and parameters: {'xgb_max_depth': 68, 'xgb_n_estimators': 35, 'xgb_learning_rate': 0.14330156317397827}. Best is trial 4 with value: 0.3315474268629703.
# [I 2023-01-09 20:07:40,244] Trial 28 finished with value: 0.7303144478005389 and parameters: {'xgb_max_depth': 2, 'xgb_n_estimators': 68, 'xgb_learning_rate': 0.04772053034098116}. Best is trial 4 with value: 0.3315474268629703.
# [I 2023-01-09 20:45:11,886] Trial 29 finished with value: 0.545893188490824 and parameters: {'xgb_max_depth': 73, 'xgb_n_estimators': 159, 'xgb_learning_rate': 0.008085565086087463}. Best is trial 4 with value: 0.3315474268629703.
# [I 2023-01-09 20:51:58,879] Trial 30 finished with value: 0.4269847546147385 and parameters: {'xgb_max_depth': 45, 'xgb_n_estimators': 48, 'xgb_learning_rate': 0.17778267844108683}. Best is trial 4 with value: 0.3315474268629703.
# [I 2023-01-09 21:08:36,263] Trial 31 finished with value: 0.3610724000220933 and parameters: {'xgb_max_depth': 98, 'xgb_n_estimators': 53, 'xgb_learning_rate': 0.08292953603524411}. Best is trial 4 with value: 0.3315474268629703.
# [I 2023-01-09 21:27:43,747] Trial 32 finished with value: 0.35388940483217246 and parameters: {'xgb_max_depth': 100, 'xgb_n_estimators': 60, 'xgb_learning_rate': 0.08632810626784362}. Best is trial 4 with value: 0.3315474268629703.
# [I 2023-01-09 21:40:20,672] Trial 33 finished with value: 0.3766864654452251 and parameters: {'xgb_max_depth': 55, 'xgb_n_estimators': 75, 'xgb_learning_rate': 0.29321906564835537}. Best is trial 4 with value: 0.3315474268629703.
# [I 2023-01-09 21:47:33,149] Trial 34 finished with value: 0.5692359209220754 and parameters: {'xgb_max_depth': 35, 'xgb_n_estimators': 63, 'xgb_learning_rate': 0.02285131177685398}. Best is trial 4 with value: 0.3315474268629703.
# [I 2023-01-09 21:57:40,839] Trial 35 finished with value: 0.36629967252047774 and parameters: {'xgb_max_depth': 79, 'xgb_n_estimators': 41, 'xgb_learning_rate': 0.19106633679220886}. Best is trial 4 with value: 0.3315474268629703.
# [I 2023-01-09 22:02:50,247] Trial 36 finished with value: 0.45208998665754657 and parameters: {'xgb_max_depth': 48, 'xgb_n_estimators': 33, 'xgb_learning_rate': 0.09149645439239483}. Best is trial 4 with value: 0.3315474268629703.
# [I 2023-01-09 22:27:49,501] Trial 37 finished with value: 0.3792391167287651 and parameters: {'xgb_max_depth': 83, 'xgb_n_estimators': 94, 'xgb_learning_rate': 0.04395650783421572}. Best is trial 4 with value: 0.3315474268629703.
# [I 2023-01-09 22:33:33,968] Trial 38 finished with value: 0.41513003803066745 and parameters: {'xgb_max_depth': 61, 'xgb_n_estimators': 29, 'xgb_learning_rate': 0.12311694351874511}. Best is trial 4 with value: 0.3315474268629703.
# [I 2023-01-09 22:38:56,579] Trial 39 finished with value: 1.047923298809881 and parameters: {'xgb_max_depth': 35, 'xgb_n_estimators': 47, 'xgb_learning_rate': 0.001882059152543855}. Best is trial 4 with value: 0.3315474268629703.
# [I 2023-01-09 22:43:17,607] Trial 40 finished with value: 1.081774464386396 and parameters: {'xgb_max_depth': 22, 'xgb_n_estimators': 60, 'xgb_learning_rate': 0.000769502090767137}. Best is trial 4 with value: 0.3315474268629703.
# [I 2023-01-09 23:01:11,277] Trial 41 finished with value: 0.35805162441569893 and parameters: {'xgb_max_depth': 100, 'xgb_n_estimators': 56, 'xgb_learning_rate': 0.08302465209263499}. Best is trial 4 with value: 0.3315474268629703.
# [I 2023-01-09 23:16:05,015] Trial 42 finished with value: 0.39646101169320136 and parameters: {'xgb_max_depth': 81, 'xgb_n_estimators': 57, 'xgb_learning_rate': 0.0562048288495509}. Best is trial 4 with value: 0.3315474268629703.
# [I 2023-01-09 23:34:47,829] Trial 43 finished with value: 0.35772952575086453 and parameters: {'xgb_max_depth': 80, 'xgb_n_estimators': 76, 'xgb_learning_rate': 0.15306753643010512}. Best is trial 4 with value: 0.3315474268629703.
# [I 2023-01-09 23:35:56,231] Trial 44 finished with value: 0.6329233583695876 and parameters: {'xgb_max_depth': 3, 'xgb_n_estimators': 76, 'xgb_learning_rate': 0.21294495388990023}. Best is trial 4 with value: 0.3315474268629703.

# [I 2023-01-10 00:04:45,348] Trial 45 finished with value: 0.331089818126148 and parameters: {'xgb_max_depth': 79, 'xgb_n_estimators': 121, 'xgb_learning_rate': 0.29482990714644686}. Best is trial 45 with value: 0.331089818126148.

# [I 2023-01-10 00:19:01,066] Trial 46 finished with value: 0.3942065908845047 and parameters: {'xgb_max_depth': 40, 'xgb_n_estimators': 116, 'xgb_learning_rate': 0.28041457760140953}. Best is trial 45 with value: 0.331089818126148.
# [I 2023-01-10 00:24:23,818] Trial 47 finished with value: 0.5777189093612596 and parameters: {'xgb_max_depth': 8, 'xgb_n_estimators': 188, 'xgb_learning_rate': 0.11711078666245668}. Best is trial 45 with value: 0.331089818126148.
# [I 2023-01-10 00:52:16,474] Trial 48 finished with value: 0.3564185399803085 and parameters: {'xgb_max_depth': 56, 'xgb_n_estimators': 164, 'xgb_learning_rate': 0.21111334819975824}. Best is trial 45 with value: 0.331089818126148.
# [I 2023-01-10 01:14:33,641] Trial 49 finished with value: 1.1047554092970044 and parameters: {'xgb_max_depth': 55, 'xgb_n_estimators': 127, 'xgb_learning_rate': 0.00012285366728221968}. Best is trial 45 with value: 0.331089818126148.
# [I 2023-01-10 01:48:09,605] Trial 50 finished with value: 0.3433488049704119 and parameters: {'xgb_max_depth': 67, 'xgb_n_estimators': 166, 'xgb_learning_rate': 0.2019949157398018}. Best is trial 45 with value: 0.331089818126148.
# [I 2023-01-10 02:20:11,082] Trial 51 finished with value: 0.34485144991757233 and parameters: {'xgb_max_depth': 67, 'xgb_n_estimators': 158, 'xgb_learning_rate': 0.19715554951490705}. Best is trial 45 with value: 0.331089818126148.
# [I 2023-01-10 02:59:33,386] Trial 52 finished with value: 0.3437270533529196 and parameters: {'xgb_max_depth': 88, 'xgb_n_estimators': 147, 'xgb_learning_rate': 0.10917943534136136}. Best is trial 45 with value: 0.331089818126148.
# [I 2023-01-10 03:29:16,314] Trial 53 finished with value: 0.35510516959596794 and parameters: {'xgb_max_depth': 68, 'xgb_n_estimators': 144, 'xgb_learning_rate': 0.15688717300970711}. Best is trial 45 with value: 0.331089818126148.
# [I 2023-01-10 04:14:58,109] Trial 54 finished with value: 0.34001208231334534 and parameters: {'xgb_max_depth': 88, 'xgb_n_estimators': 171, 'xgb_learning_rate': 0.11475271448154668}. Best is trial 45 with value: 0.331089818126148.
# [I 2023-01-10 04:39:07,634] Trial 55 finished with value: 0.3767941875640358 and parameters: {'xgb_max_depth': 46, 'xgb_n_estimators': 172, 'xgb_learning_rate': 0.20679235419800082}. Best is trial 45 with value: 0.331089818126148.
# [I 2023-01-10 05:18:47,086] Trial 56 finished with value: 0.35517993410768095 and parameters: {'xgb_max_depth': 85, 'xgb_n_estimators': 151, 'xgb_learning_rate': 0.0657734220438489}. Best is trial 45 with value: 0.331089818126148.
# [I 2023-01-10 06:00:04,730] Trial 57 finished with value: 0.41505165026413543 and parameters: {'xgb_max_depth': 65, 'xgb_n_estimators': 198, 'xgb_learning_rate': 0.017005053827748022}. Best is trial 45 with value: 0.331089818126148.
# [I 2023-01-10 06:12:49,088] Trial 58 finished with value: 0.48633790305871544 and parameters: {'xgb_max_depth': 32, 'xgb_n_estimators': 125, 'xgb_learning_rate': 0.030487535187413588}. Best is trial 45 with value: 0.331089818126148.
# [I 2023-01-10 06:41:34,076] Trial 59 finished with value: 0.33296356215381034 and parameters: {'xgb_max_depth': 73, 'xgb_n_estimators': 130, 'xgb_learning_rate': 0.2937688619297414}. Best is trial 45 with value: 0.331089818126148.


# +
# Even with a reduced search space, search and fitting an xgb was unacceptably slow, on the order of ~20 hours
# execution queued 13:55:18 2023-01-09 

# Due to time constraints, I'm using the best hps for 2021 for all years. 

# +
# Setup ----------------------------------------------------------------------
trial_name = 'xgb'


n_trials= 60 #120  #FIXME
n_jobs = 1
n_jobs_xgb = 20
n_jobs_retrain = 1
# test_this_year = '2014'

downsample = False# FIXME
downsample_train = 1000
downsample_test  =  100

reset_trial_name = trial_name

if True == False:
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

        print("""
    ------------------------------------------
        """)
        def fit_single_rep(Rep = 1):
            # Fit Best HPS --------------------------------------------------------------- 
            cache_save_name = cache_path+trial_name+'_'+str(Rep)+'_mod.pkl'

            # Load cached model if it exists
            if os.path.exists(cache_save_name):
                xgb = pkl.load(open(cache_save_name, 'rb'))   
            else:
    #             regr = XGBRegressor(
    #                     max_depth = study.best_trial.params['xgb_max_depth'], 
    #                     n_estimators = study.best_trial.params['xgb_n_estimators'],
    #                     learning_rate = study.best_trial.params['xgb_learning_rate'],
    #                     objective='reg:squarederror'
    #                     )
                regr = XGBRegressor(
                        max_depth = 79,                      # These values come 
                        n_estimators = 121,                  # from the best trial
                        learning_rate = 0.29482990714644686, # on 2021 data
                        objective='reg:squarederror'
                        )            
                xgb = regr.fit(train_x_2d, train_y_1d)
                # save    
                pkl.dump(xgb, open(cache_save_name, 'wb'))   

            # Record Predictions -----------------------------------------------------
            out = phno.copy()
            out['YHat'] = xgb.predict(full_x_2d)
            out['YMat'] = YMat
            out['Y_center'] = YMat_center
            out['Y_scale'] = YMat_scale
            out['Class'] = trial_name
            out['CV'] = test_this_year
            out['Rep'] = Rep
            out.to_csv('./notebook_artifacts/4_modeling_xgb/'+trial_name+'_'+str(Rep)+'xgbYHats.csv')
    #         out.to_csv('./data/Shared_Model_Output/'+trial_name+'_'+str(Rep)+'xgbYHats.csv')

        # use joblib to get replicate models all at once
    #     Parallel(n_jobs=n_jobs_retrain)(delayed(fit_single_rep)(Rep = i) for i in range(10))
        for i in tqdm.tqdm(range(10)):
            fit_single_rep(Rep = i)

    # executed in 1d 3h 18m, finished 18:34:21 2023-01-11

# +
# # copy from ./notebook_artifacts/4_modeling_xgb/ to ./data/Shared_Model_Output/

if True == False:
    import os, re
    import shutil

    start_path = './notebook_artifacts/4_modeling_xgb/'
    end_path   = './data/Shared_Model_Output/'

    yHat_files = [e for e in os.listdir(start_path) if re.match('xgb\d+_\dxgbYHats.csv$', e)]

    for yHat_file in yHat_files:
        shutil.move(
            start_path+yHat_file,
            end_path+yHat_file
        )
