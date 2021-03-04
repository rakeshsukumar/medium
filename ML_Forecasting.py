# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 19:28:46 2020

"""

import time
import numpy as np
import pandas as pd
import sklearn as sk
import math
import os
import csv

import tqdm
import itertools
import os

from pandas import Series
from pandas import DataFrame
from pandas import concat

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble.forest import RandomForestRegressor

from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_regression

from sklearn.preprocessing import StandardScaler
from sklearn import neighbors

from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd
from sklearn.model_selection import cross_val_score

pd.set_option('display.max_rows', 120)
#dir_path = os.path.dirname(os.path.realpath(__file__))

import os
import pandas as pd
import multiprocessing as mp
import os
import concurrent.futures

import sys
import logging
import time


def logging_(path):   
    logging.getLogger('fbprophet').setLevel(logging.WARNING)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, filename= path, filemode = 'w', format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')


def HADS_data(path):
    df = pd.read_csv(path, delimiter= ',')
    df['TRDNG_WK_END_DT'] = pd.to_datetime(df['TRDNG_WK_END_DT'], infer_datetime_format = True)
    gb = df.groupby(['territory', 'KEY'])
    dataframes = [gb.get_group(x) for x in gb.groups]
    return df , dataframes


def LADS_data(path):
    df = pd.read_csv(path, delimiter= ',')
    df['TRDNG_WK_END_DT'] = pd.to_datetime(df['TRDNG_WK_END_DT'], infer_datetime_format = True)
    gb = df.groupby(['territory', 'KEY'])
    dataframes = [gb.get_group(x) for x in gb.groups]
    return df , dataframes





def get_feature_type(featDF):
    print("[IN] function to get categorical / continuous feature list : get_feature_type")
    
    catg_features = []
    contd_features = []
    cnst_features = []
    
    featDF = pd.DataFrame(featDF.fillna(0))
    
    for col in featDF.columns:
        
        uniqueVal = len(pd.unique(featDF[[col]].values.ravel()))
        
        if (uniqueVal == 2):
            
            catg_features.append(col)
            
        elif (uniqueVal == 1):
            
            cnst_features.append(col)            
            
        else:
            
            contd_features.append(col)
    
    print("[OUT] function to get categorical / continuous feature list : get_feature_type")
    return catg_features , contd_features , cnst_features




"""
#create year/month/week index from date stamp
#create month/yearly seasonal index from historical sales
"""                                  
def season_index_features(ssn_df): 
    print("[IN] get season index on HADS data : season_index_features")
                                  
    ssn_df['TRDNG_WK_STRT_DT'] = pd.to_datetime(ssn_df['TRDNG_WK_STRT_DT'],format='%d%b%Y')
    ssn_df['TRDNG_WK_END_DT'] = pd.to_datetime(ssn_df['TRDNG_WK_END_DT'],format='%d%b%Y')

    ssn_df['YEAR_ID'] = ssn_df['TRDNG_WK_STRT_DT'].dt.year
    ssn_df['MONTH_ID'] = ssn_df['TRDNG_WK_STRT_DT'].dt.month
    ssn_df['MON_WK_ID'] = 1
    ssn_df['MON_WK_ID'] = ssn_df.groupby(['territory','KEY','YEAR_ID','MONTH_ID'])['MON_WK_ID'].apply(lambda x: x.cumsum())

    TerrCATGMonthAVG = pd.DataFrame({'month_avg' : ssn_df.groupby(['GRP_NM','DPT_NM','CLSS_NM','SUB_CLSS_NM','MONTH_ID'])['RTL_QTY'].apply(lambda x: x.mean())}).reset_index()

    TerrCATGoAVG = pd.DataFrame({'all_avg' : ssn_df.groupby(['GRP_NM','DPT_NM','CLSS_NM','SUB_CLSS_NM'])['RTL_QTY'].apply(lambda x: x.mean())}).reset_index()

    TerrCATGrAVG = ssn_df[ssn_df['YEAR_ID'] == ssn_df['YEAR_ID'].max()]
    TerrCATGrAVG = pd.DataFrame({'rcnt_avg' : TerrCATGrAVG.groupby(['GRP_NM','DPT_NM','CLSS_NM','SUB_CLSS_NM'])['RTL_QTY'].apply(lambda x: x.mean())}).reset_index()
                                
    TerrCATGMonthAVG = pd.merge(TerrCATGMonthAVG, TerrCATGoAVG, on=['GRP_NM','DPT_NM','CLSS_NM','SUB_CLSS_NM'], how='left')
    TerrCATGMonthAVG = pd.merge(TerrCATGMonthAVG, TerrCATGrAVG, on=['GRP_NM','DPT_NM','CLSS_NM','SUB_CLSS_NM'], how='left')

    TerrCATGMonthAVG['CATG_SSN_ID_O'] = TerrCATGMonthAVG['month_avg'] / TerrCATGMonthAVG['all_avg']
    TerrCATGMonthAVG['CATG_SSN_ID_R'] = TerrCATGMonthAVG['month_avg'] / TerrCATGMonthAVG['rcnt_avg']

    TerrCATGMonthAVG = TerrCATGMonthAVG.drop(['month_avg','all_avg','rcnt_avg'],1)

    #ssn_df = pd.merge(ssn_df, TerrCATGMonthAVG, on=['GRP_NM','DPT_NM','CLSS_NM','SUB_CLSS_NM','MONTH_ID'], how='left')
    
    #return ssn_df, TerrCATGMonthAVG
    print("[OUT] get season index on HADS data : season_index_features")
    return TerrCATGMonthAVG

"""
create date since last event feature for binary event variables
"""
def days_since_last_event(dsle_df, dvar_list,primary_key):
    print("[IN] create days since last event features : days_since_last_event")
    
    dsle_df = dsle_df.sort_values(by=primary_key)
    
    for dvar in dvar_list:
    
        if dvar in dsle_df.columns:
            
            dsle_var = (dvar + '_' + 'DSLE')
            
            dsle_df['dvar_dummy1'] = dsle_df.groupby(['territory','KEY'])[dvar].apply(lambda x: x != x.shift(1)).cumsum()
            dsle_df['dvar_dummy'] = dsle_df.groupby(['territory','KEY','dvar_dummy1']).cumcount()+1
            
            dsle_df[dsle_var] = dsle_df['dvar_dummy']*7
            dsle_df.loc[(dsle_df[dvar] == 1) | (dsle_df[dsle_var] > 60), [dsle_var]] = 0

            dsle_df = dsle_df.drop(['dvar_dummy1','dvar_dummy'],1)
    
    print("[OUT] create days since last event features : days_since_last_event")
    return dsle_df

"""
create lagged terms for binary variables mostly seasonal / promotional event
"""
def binary_lag_features(bvar_df, bvar_list,primary_key):
    print("[IN] create binary lag terms of event features (eg. Ramadan) : binary_lag_features")
    
    bvar_df = bvar_df.sort_values(by=primary_key)
    
    for bvar in bvar_list:
    
        if bvar in bvar_df.columns:
    
            bvar_df['bvar_dummy1'] = bvar_df.groupby(['territory','KEY'])[bvar].apply(lambda x: x != x.shift(1)).cumsum()        
            bvar_df['bvar_dummy'] = bvar_df.groupby(['territory','KEY','bvar_dummy1']).cumcount()+1

            bvar_df.loc[bvar_df[bvar] == 0, 'bvar_dummy'] = 0

            bvar_dummy = bvar_df.loc[bvar_df['bvar_dummy'] != 0, 'bvar_dummy'].unique()
    
            for lag_term in bvar_dummy:
        
                lagged_bvar = (bvar + '_' + str(lag_term))
        
                bvar_df[lagged_bvar] = bvar_df['bvar_dummy'].apply(lambda x: 1 if x == lag_term else 0)
            
            #bvar_df = bvar_df.drop(['bvar_dummy1','bvar_dummy',bvar],1)
            bvar_df = bvar_df.drop(['bvar_dummy1','bvar_dummy'],1)
    
    print("[OUT] create binary lag terms of event features (eg. Ramadan) : binary_lag_features")
    return bvar_df
    
"""
create lagged terms for continuos variables
"""    
def continuous_lag_features(cvar_df, cvar_list, cvar_lag,primary_key):
    print("[IN] create normal lag terms of features (eg. DISC_PER) : continuous_lag_features")
    
    cvar_df = cvar_df.sort_values(by=primary_key)
    
    for cvar in cvar_list:
        
        if cvar in cvar_df.columns:
    
            for lag_term in range(cvar_lag):

                lagged_cvar = (cvar + '_' + str(lag_term+1))
        
                cvar_df[lagged_cvar] = cvar_df.groupby(['territory','KEY'])[cvar].shift(lag_term+1)
                
    print("[OUT] create normal lag terms of features (eg. DISC_PER) : continuous_lag_features")
    return cvar_df

"""
create fourier term (sine + consine) as independent variable
"""                                  
def fourier_term_features(fft_df, fterm,primary_key):
    print("[IN] create fourier term features : fourier_term_features")
    
    fft_df = fft_df.sort_values(by=primary_key)
    
    fft_df['T_index'] = 1
    fft_df['T_index'] = fft_df.groupby(['territory','KEY'])['T_index'].cumsum()
    
    for i in range(fterm):
        
        for j in range(fterm):
            
            ft_var1 = ('FT_S' + str(i+1) + str(j+1))
            ft_var2 = ('FT_C' + str(i+1) + str(j+1))
            ft_var = ('FT_' + str(i+1) + str(j+1))
            
            fft_df[ft_var1] = np.sin((2*np.pi*(i+1)*fft_df['T_index'])/52)
            fft_df[ft_var2] = np.cos((2*np.pi*(j+1)*fft_df['T_index'])/52)
            fft_df[ft_var] = fft_df[ft_var1] + fft_df[ft_var2]

            fft_df = fft_df.drop([ft_var1,ft_var2],1)
            
    fft_df = fft_df.drop('T_index',1)

    print("[OUT] create fourier term features : fourier_term_features")
    return fft_df

"""
create time series lag term and lag avg. category sales ratio as features
"""                                  
def ts_lag_features(lag_df, lag_range,primary_key):
    print("[IN] create time series lag term features : ts_lag_features")
    
    lag_df = lag_df.sort_values(by=primary_key)
    
    for i in range(lag_range):
        
        lag_col = ('RTL_LAG' + str(i+1))
        catg_sls_col = ('CATG_SLS_RATIO' + str(i+1))
        
        lag_df[lag_col] = lag_df.groupby(['territory','KEY'])['RTL_QTY'].shift(i+1)
        
        #TerrCATGwkSLS = pd.DataFrame({'CATG_RTL_QTY' : lag_df.groupby(['territory','GRP_NM','DPT_NM','CLSS_NM','SUB_CLSS_NM','YEAR_ID','MONTH_ID','MON_WK_ID'])[lag_col].apply(lambda x: x.sum())}).reset_index()
        
        #lag_df = pd.merge(lag_df, TerrCATGwkSLS, on=['territory','GRP_NM','DPT_NM','CLSS_NM','SUB_CLSS_NM','YEAR_ID','MONTH_ID','MON_WK_ID'], how='left')
        #lag_df[catg_sls_col] = lag_df[lag_col] / lag_df['CATG_RTL_QTY']
    
        #lag_df = lag_df.drop('CATG_RTL_QTY',1)
    
    print("[OUT] create time series lag term features : ts_lag_features")
    return lag_df
    
"""
create basic stats ie. min/max/mean of the lag term as features
"""                                  
def lag_stat_features(lag_df, lag_range,primary_key):
    print("[IN] create ts lag term MIN/MAX/MEAN features : lag_stat_features")
    
    lag_df = lag_df.sort_values(by=primary_key)

    for j in range(lag_range-1):
    
        lag_min = ('RTL_LAG_MIN' + str(j+1))
        lag_max = ('RTL_LAG_MAX' + str(j+1))
        lag_mean = ('RTL_LAG_MEAN' + str(j+1))
#    mean_col_base = inputDFforecast.columns.get_loc('RTL_QTY')
        mean_col_strt = lag_df.columns.get_loc('RTL_LAG' + str(1))
        mean_col_end = lag_df.columns.get_loc('RTL_LAG' + str(j+2))+1
        mean_col = list(range(mean_col_strt, mean_col_end))
#    mean_col.append(mean_col_base)
        lag_df[lag_min] = lag_df.iloc[:,mean_col].min(axis=1, skipna=0)
        lag_df[lag_max] = lag_df.iloc[:,mean_col].max(axis=1, skipna=0)
        lag_df[lag_mean] = lag_df.iloc[:,mean_col].mean(axis=1, skipna=0)
        print("[OUT] create ts lag term MIN/MAX/MEAN features : lag_stat_features")
    return lag_df


"""
Pearson Correlation based feature selection
"""                     
#CORR_feature_selection(impFS_X, impFS_Y, all_var_list)     
#corrFS_X, corrFS_Y, corrFS_list = impFS_X, impFS_Y, all_var_list         
def CORR_feature_selection(corrFS_X, corrFS_Y, corrFS_list):
    print("[IN] pearson correlation feature selection : CORR_feature_selection")
    
    corrFS_X = pd.DataFrame(corrFS_X.fillna(0))
    
    vars_0dev = corrFS_X.loc[:, corrFS_X.std() == 0].columns.values
    
    print(vars_0dev)
    
    corrFS_list = list(set(corrFS_list) - set(vars_0dev))
    #Editing the filteration
    corrFS_X = corrFS_X.loc[:, [i for i in corrFS_list if i in corrFS_X.columns]]
    
    corrFS_sel = []
    # calculate the correlation with y for each feature
    for i in corrFS_X.columns.tolist():
        print(i)    
        corrFS = np.corrcoef(corrFS_X[i], corrFS_Y.values.ravel())[0, 1]
        corrFS_sel.append(corrFS)
        
    # replace NaN with 0
    corrFS_sel = [0 if np.isnan(i) else i for i in corrFS_sel]
    
    # get feature name
    corr_feature = corrFS_X.iloc[:,np.argsort(np.abs(corrFS_sel))[-125:]].columns.tolist()
    
    print(corr_feature)
    print("[OUT] pearson correlation feature selection : CORR_feature_selection")
    return corr_feature

"""
Random Forest (RF) based feature selection
"""                                  
#rfFS_X, rfFS_Y, rfFS_list=impFS_X, impFS_Y, CORR_feature_list

def RF_feature_selection(rfFS_X, rfFS_Y, rfFS_list):
    print("[IN] random forest feature selection : RF_feature_selection")

    rfFS_X = pd.DataFrame(rfFS_X.fillna(0))
    
    #Test_y_rf = pd.DataFrame([1]*217,columns = ['RTL_QTY'])
    #rfFS_Y = Test_y_rf
    rfFS_Y = pd.DataFrame(rfFS_Y.fillna(method = 'backfill'))
    rf_obj = RandomForestRegressor(n_estimators = 500, bootstrap = True, max_features = 'auto',random_state=0)
    #rfFS_model = SelectFromModel(rfc_obj, threshold=0.01, max_features=25)
    rfFS_model = SelectFromModel(rf_obj)        
    rfFS_model.fit(rfFS_X, rfFS_Y.values.ravel())
    rf_feature = rfFS_X.columns[(rfFS_model.get_support())]

    print(rf_feature)
    print("[OUT] random forest feature selection : RF_feature_selection")
    return rf_feature

"""
Lasso Regression (L1 regularisation) based feature selection
"""                                  
def LASSO_feature_selection(lasoFS_X, lasoFS_Y, lasoFS_list):
    print("[IN] lasso regression feature selection : LASSO_feature_selection")
    
    lasoFS_X = pd.DataFrame(lasoFS_X.fillna(0))
    
    #laso_obj = Lasso(alpha=0.5)
    laso_obj = LassoCV(cv=3)
    
    laso_model = SelectFromModel(laso_obj, threshold=0.5)
    
    laso_model.fit(lasoFS_X, lasoFS_Y.values.ravel())
    
    laso_feature = lasoFS_X.columns[(laso_model.get_support())]
    
    n_feature = len(laso_feature)
    
    while n_feature > 15:
        
        laso_model.threshold += 0.05
        laso_feature = lasoFS_X.columns[(laso_model.get_support())]
        n_feature = len(laso_feature)

    print(laso_feature)
    print("[OUT] lasso regression feature selection : LASSO_feature_selection")
    return laso_feature

"""
Random Forest regression for time series forecast / prediction
"""             
#RFT_trainX, RFT_trainY, RFT_predX, RFT_feature, RFTitr, RFTnbr,histDF_ip=TS_train_X, TS_train_Y, TS_pred_X, featList, ld, 1,histDF
def RFtree_TS_forecast(RFT_trainX, RFT_trainY, RFT_predX, RFT_feature, RFTitr, RFTnbr,histDF_ip):
    
    if (RFTitr == 0):
        #start_time_rf_training=time.time()
        # DF for hyperparameter tuning with cross-validation
        RFT_trainCV_X = pd.DataFrame(RFT_trainX)
        RFT_trainCV_Y = pd.DataFrame(RFT_trainY)
        
        # number of trees in random forest
        #n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 25)]
        # number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
        # method of selecting samples for training each tree
        bootstrap = [True, False]
        
        # create the random grid for randomized search
        RFT_RSgrid = {
                    #'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'bootstrap': bootstrap}
        
        # use random grid to search for best hyperparameter range (not using n_jobs)
        RFT_RSmodel = RandomForestRegressor()
        # random search with cross validation
        RFT_randomCV = RandomizedSearchCV(estimator = RFT_RSmodel, 
                                       param_distributions = RFT_RSgrid, 
                                       n_iter = 100, 
                                       cv = 5, verbose=2, random_state=42, n_jobs = -1
                                       ,scoring = 'neg_median_absolute_error'
                                       )
         ###########################
        
        # fit random search model with training DF
        RFT_randomCV.fit(RFT_trainCV_X, RFT_trainCV_Y)
        #end_time_rf_hparam_tuning = time.time()
        #duration_rf_hparam_tuning =  end_time_rf_hparam_tuning - start_time_rf_training  
        
        
        
        
        # optimal / tunned hyperparameter set
        RFTnbr_opt = RFT_randomCV.best_params_
        
        #RFTest_parm = list(RFTnbr_opt.values())[0]
        RFTfeat_parm = list(RFTnbr_opt.values())[0]
        RFTdpth_parm = list(RFTnbr_opt.values())[1]
        RFTboot_parm = list(RFTnbr_opt.values())[2]
        
        #start_time_rf_model_fitting=time.time()
        # random forest regressor instance with tunned hyperparameters
        RFT_model = RandomForestRegressor(
                                        #n_estimators = 500,
                                          max_depth = RFTdpth_parm,
                                          max_features = RFTfeat_parm,
                                          bootstrap = RFTboot_parm,
                                          verbose=2, random_state=42, n_jobs = -1)
        
        #cv_out_rf_rs=cross_val_score(RFT_model, RFT_trainCV_X, RFT_trainCV_Y, cv = 5,scoring = 'neg_mean_absolute_error')
        #neg_mean_absolute_error
        #score_rf_test = np.median(cv_out_rf_rs)
        # fit random forest model on training set
        RFT_model.fit(RFT_trainX , RFT_trainY)

        
        RFT_feat_imp=pd.DataFrame(RFT_trainX.columns,
                                  RFT_model.feature_importances_).reset_index().rename(columns={'index':'RF_Feature_Importance',
                                                                                                      0:'Features_RF'}).loc[:,['Features_RF','RF_Feature_Importance']]
        # predict using model object and prediction DF
        RFT_forecast = RFT_model.predict(RFT_predX)[0]
        
        #import math
        #RFT_forecast=math.ceil(RFT_forecast)
        
        histDF_ip_ss = histDF_ip.loc[:,['territory','KEY','TRDNG_WK_STRT_DT','TRDNG_WK_END_DT','RTL_QTY']]
        histDF_ip_ss['HADS_Prediction_RF'] = [i for i in RFT_model.predict(RFT_trainX)]
        
        print(RFT_forecast)
        
    else:
        ###Control never comes here
        #RFTest_parm = list(RFTnbr.values())[0]
        RFTfeat_parm = list(RFTnbr.values())[0]
        RFTdpth_parm = list(RFTnbr.values())[1]
        RFTboot_parm = list(RFTnbr.values())[2]
        
        # random forest regressor instance with tunned hyperparameters
        RFT_model = RandomForestRegressor(
            #n_estimators = 500,
                                          max_depth = RFTdpth_parm,
                                          max_features = RFTfeat_parm,
                                          bootstrap = RFTboot_parm,
                                          verbose=2, random_state=42, n_jobs = -1)
        # fit random forest model on training set
        RFT_model.fit(RFT_trainX , RFT_trainY)
        # predict using model object and prediction DF
        RFT_forecast = RFT_model.predict(RFT_predX)[0]        
        
        import math
        #RFT_forecast=math.ceil(RFT_forecast)
        
        #if RFT_forecast < 0:
        #    RFT_forecast = 0
        
        histDF_ip_ss = histDF_ip.loc[:,['territory','KEY','TRDNG_WK_STRT_DT','TRDNG_WK_END_DT','RTL_QTY']]
        histDF_ip_ss['HADS_Prediction_RF'] = [i for i in RFT_model.predict(RFT_trainX)]
        
        RFTnbr_opt = RFTnbr
        
        print(RFT_forecast)

    return RFT_forecast , RFTnbr_opt , RFT_model, histDF_ip_ss,RFT_feat_imp

"""
KNN regression for time series forecast / prediction
"""                                  
#KNN_trainX, KNN_trainY, KNN_predX, KNN_feature, KNNitr, Knbr,histDF_ip = TS_train_X, TS_train_Y, TS_pred_X, featList, ld, 1,histDF
#forecast_val , param_val, model_val, hads_prediction= KNN_TS_forecast(TS_train_X, TS_train_Y, TS_pred_X, featList, ld, 1,histDF)
#KNN_trainX, KNN_trainY, KNN_predX, KNN_feature, KNNitr, Knbr,histDF_ip = TS_train_X, TS_train_Y, TS_pred_X, featList, ld, 1,histDF
def KNN_TS_forecast(KNN_trainX, KNN_trainY, KNN_predX, KNN_feature, KNNitr, Knbr,histDF_ip):
        
    if (KNNitr == 0):
        
        KNN_trainCV_X = KNN_trainX.iloc[:-13,:]
        KNN_trainCV_Y = KNN_trainY.iloc[:-13,:]
        
        KNN_testCV_X = KNN_trainX.tail(13)
        KNN_testCV_Y = KNN_trainY.tail(13)
                
        KNNparam = {'n_neighbors':[2,3,4,5,6,7,8,9,10]}
        
        KNN_CVmodel = neighbors.KNeighborsRegressor()
        
        KNNgrid = GridSearchCV(KNN_CVmodel, KNNparam, cv=5)
        
        KNNgrid.fit(KNN_trainX , KNN_trainY)
        Knbr_opt = list(KNNgrid.best_params_.values())[0]
        
        KNN_model = neighbors.KNeighborsRegressor(n_neighbors = Knbr_opt)
        
        KNN_model.fit(KNN_trainX , KNN_trainY)
        
        #import math
        
        KNN_forecast = KNN_model.predict(KNN_predX)[0][0]
        
        #if KNN_forecast < 0:
         #   KNN_forecast = 0
        
        
        #HADS_Prediction=pd.DataFrame(KNN_model.predict(KNN_trainX),columns = ['prediction_hads_knn'])
        #['territory','KEY','TRDNG_WK_STRT_DT','TRDNG_WK_END_DT']
        histDF_ip_ss = histDF_ip.loc[:,['territory','KEY','TRDNG_WK_STRT_DT','TRDNG_WK_END_DT']]
        histDF_ip_ss['HADS_Prediction_KNN'] = KNN_model.predict(KNN_trainX)
        
        
    else:
        
        KNN_model = neighbors.KNeighborsRegressor(n_neighbors = Knbr)
        
        KNN_model.fit(KNN_trainX , KNN_trainY)
        
        KNN_forecast = KNN_model.predict(KNN_predX)[0][0]
        
        #import math
        #KNN_forecast=math.ceil(KNN_forecast)
        
        #if KNN_forecast < 0:
        #   KNN_forecast = 0
        
        
        histDF_ip_ss = histDF_ip.loc[:,['territory','KEY','TRDNG_WK_STRT_DT','TRDNG_WK_END_DT']]
        histDF_ip_ss['HADS_Prediction_KNN'] = KNN_model.predict(KNN_trainX)
        
        Knbr_opt = Knbr

    return KNN_forecast , Knbr_opt , KNN_model, histDF_ip_ss

"""
Stepwise regression for time series forecast / prediction
"""                                  
#LR_trainX, LR_trainY, LR_predX, LR_feature, LRitr,histDF_ip = TS_train_X, TS_train_Y, TS_pred_X, featList, ld,histDF
def LREG_TS_forecast(LR_trainX, LR_trainY, LR_predX, LR_feature, LRitr,histDF_ip):
            
    if (LRitr == 0):
        
        LRmodel = LinearRegression()
        
        LRmodel.fit(LR_trainX, LR_trainY)
        
        #Getting Model coefficients####
        cdf = pd.DataFrame(LRmodel.coef_[0], 
                           [LR_trainX.columns.tolist()], 
                           columns=['LREG_New_Coefficients']).reset_index().rename(columns={'level_0':'Columns'})
        int_ = pd.DataFrame(LRmodel.intercept_[0],['Intercept'],
                            columns=['LREG_New_Coefficients']).reset_index().rename(columns = {'index':'Columns'})
        
        cdf_all=cdf.append(int_)
        
        
        LR_forecast = LRmodel.predict(LR_predX)[0][0]
        
        #import math
        
        #LR_forecast=math.ceil(LR_forecast)
        
        #if LR_forecast < 0:
         #   LR_forecast = 0
        
        histDF_ip_ss = histDF_ip.loc[:,['territory','KEY','TRDNG_WK_STRT_DT','TRDNG_WK_END_DT']]
        histDF_ip_ss['HADS_Prediction_LREG'] = LRmodel.predict(LR_trainX)
    
    else:
                
        LRmodel = LinearRegression()
        
        LRmodel.fit(LR_trainX, LR_trainY)
        
        LR_forecast = LRmodel.predict(LR_predX)[0][0]
        
        #LR_forecast=math.ceil(LR_forecast)
        
        #if LR_forecast < 0:
         #   LR_forecast = 0
        
        histDF_ip_ss = histDF_ip.loc[:,['territory','KEY','TRDNG_WK_STRT_DT','TRDNG_WK_END_DT']]
        
        histDF_ip_ss['HADS_Prediction_LREG'] = LRmodel.predict(LR_trainX)

    return LR_forecast , LRmodel, histDF_ip_ss,cdf_all

"""
does forecast for a selected method one row at a time (ie. iterative)
"""      
#histDF, leadDF, leadWK, featList, modelNM,params_feat_input,primary_key = hist_df, lead_df, params_feat_input[4], feature_list_final, 'LREG',params_feat_input,primary_key
def forecast_iteration(histDF, leadDF, leadWK, featList, modelNM,params_feat_input,primary_key):
        
    DF_pred_fcst = pd.DataFrame()
    
        
    for ld in range(leadWK):
                
        if (ld == 0):
            histDF_new = histDF.append(leadDF[(leadDF['ROW_NUM'] == ld)].drop('ROW_NUM',1)).sort_values(by=['TRDNG_WK_END_DT'])
            fcst_df_itr = pd.DataFrame(histDF_new)
            
            fcst_df_itr = ts_lag_features(fcst_df_itr, params_feat_input[2],primary_key)
            fcst_df_itr = lag_stat_features(fcst_df_itr, params_feat_input[2],primary_key)
            #fcst_df_itr = bng_band_features(fcst_df_itr, lag)
            
            fcst_df_itr = fcst_df_itr.drop(['level_0','index'],1)
            ##Missing Value treatment to be done
            
            non_char_cols = fcst_df_itr.select_dtypes(['int64','int32','float64','float32']).columns
            char_cols = fcst_df_itr.select_dtypes(['object','datetime64[ns]']).columns
            
            
            #non_char_cols = pd.concat()
            fcst_df_itr_num=fcst_df_itr.loc[:,[i for i in fcst_df_itr.columns if i in non_char_cols]]
            
            ###Using Polynomial order 2 interpolation
            #astype(float)
            #fcst_df_itr.select_dtypes(['object']).columns
            fcst_df_itr_num = fcst_df_itr_num.interpolate(method = 'linear',axis=0,limit_direction = 'both')
            
            fcst_df_itr=pd.concat([fcst_df_itr_num,fcst_df_itr.loc[:,char_cols]],axis=1)
            
            # get all columns needed for TS iterative lead period forecast
            model_features = list(primary_key) + list(['RTL_QTY']) + list(featList)
                        
            # subset forecast DF for columns required for modelling
            #Editing the column filters due to error
            fcst_df_itr = fcst_df_itr.loc[:,[i for i in model_features if i in fcst_df_itr.columns or i in primary_key]]
            fcst_df_itr = pd.DataFrame(fcst_df_itr)
            
            # create training and prediction DF
            # only 1 row would be predicted in each iteration
            TS_train_DF = pd.DataFrame(fcst_df_itr.iloc[:-1,:])
            TS_pred_DF = pd.DataFrame(fcst_df_itr.tail(1))
            
            # create _X (feature set) & _Y (target set) for Train DF
            TS_train_Y = TS_train_DF.loc[-TS_train_DF.isnull().any(axis=1),['RTL_QTY']]
            #Editing the column filters due to error
            TS_train_X = TS_train_DF.loc[-TS_train_DF.isnull().any(axis=1), [i for i in featList if i in TS_train_DF.columns]]
            
            print(TS_train_Y.shape)
            print(TS_train_X.columns)
            # get feature type ie. categorical | continuous | constant
            catgFS_new , contdFS_new , cnstFS_new = get_feature_type(TS_train_X)
                        
            # standardizing training DF & prediction DF
            TSscaler = StandardScaler()
            if(len(contdFS_new)>0):
                TSscaler.fit(TS_train_X.loc[:,contdFS_new])
                # use scaler object to transform train and pred data for standardization
                TS_train_X1 = TSscaler.transform(TS_train_X.loc[:,contdFS_new])
                TS_train_X1 = pd.DataFrame(TS_train_X1 , columns = TS_train_X.loc[:,contdFS_new].columns)
                TS_train_X = pd.merge(TS_train_X1.reset_index(drop=True) , TS_train_X.loc[:,catgFS_new].reset_index(drop=True) , how='inner' , left_index = True , right_index = True)
                print(TS_train_X.columns)
                # transform dataframe into numpy ndarray
                #TS_train_X = TS_train_X.as_matrix()
                # use scaler object to transform pred data for standardization
                TS_pred_X1 = TSscaler.transform(TS_pred_DF.loc[:,contdFS_new])
                TS_pred_X1 = pd.DataFrame(TS_pred_X1 , columns = TS_pred_DF.loc[:,contdFS_new].columns)
                TS_pred_X = pd.merge(TS_pred_X1.reset_index(drop=True) , TS_pred_DF.loc[:,catgFS_new].reset_index(drop=True) , how='inner' , left_index = True , right_index = True)
                print(TS_pred_X.columns)
                #chk = 1
            else:
                #chk = 2
                TS_train_X =TS_train_X.loc[:,catgFS_new].reset_index(drop=True)
                TS_pred_X = TS_pred_DF.loc[:,catgFS_new]
            # transform dataframe into numpy ndarray
            #TS_pred_X = TS_pred_X.as_matrix()
            
            forecast_val = (modelNM + '_fcst')
            param_val = (modelNM + 'param')
            model_val = (modelNM + 'model')
            
            if (modelNM == 'KNN'):                        
                # using KNN for time series forecast / prediction
                cdf_all = 'KNN_All'
                forecast_val , param_val, model_val, hads_prediction= KNN_TS_forecast(TS_train_X, TS_train_Y, TS_pred_X, featList, ld, 1,histDF)
                histDF_new.iloc[-1 , histDF_new.columns.get_loc('RTL_QTY')] = forecast_val
            
            elif (modelNM == 'LREG'):            
                # using Linear Regression for time series forecast / prediction
                forecast_val , model_val, hads_prediction, cdf_all = LREG_TS_forecast(TS_train_X, TS_train_Y, TS_pred_X, featList, ld,histDF)
                histDF_new.iloc[-1 , histDF_new.columns.get_loc('RTL_QTY')] = forecast_val
                
            elif (modelNM == 'RFT'):
                #cdf_all = 'All'
                # using Random Forest for time series forecast / prediction
                forecast_val , param_val, model_val, hads_prediction,cdf_all = RFtree_TS_forecast(TS_train_X, TS_train_Y, TS_pred_X, featList, ld, 1,histDF)
                histDF_new.iloc[-1 , histDF_new.columns.get_loc('RTL_QTY')] = forecast_val
                
            else:
                print('awaiting new forecast method')
                
            fcst_col1 = (modelNM + '_FCST')
            fcst_col2 = (modelNM + '_FLG')
                        
            TS_pred_DF = TS_pred_DF.loc[:,primary_key]
            TS_pred_DF[fcst_col1] = forecast_val
            TS_pred_DF[fcst_col2] = 1
            DF_pred_fcst = DF_pred_fcst.append(TS_pred_DF)
            #Time logging

            
        else:
            
            histDF_new = histDF_new.append(leadDF[(leadDF['ROW_NUM'] == ld)].drop('ROW_NUM',1)).sort_values(by=['TRDNG_WK_END_DT'])
            fcst_df_itr = pd.DataFrame(histDF_new)
            
            fcst_df_itr = ts_lag_features(fcst_df_itr,params_feat_input[2] ,primary_key)
            fcst_df_itr = lag_stat_features(fcst_df_itr,  params_feat_input[2],primary_key)
            #fcst_df_itr = bng_band_features(fcst_df_itr, lag)
            
            fcst_df_itr = fcst_df_itr.drop(['level_0','index'],1)
                                                
            # subset forecast DF for columns required for modelling
            fcst_df_itr = fcst_df_itr.loc[:,[i for i in model_features if i in fcst_df_itr.columns]]
                        
            # create training and prediction DF
            # only 1 row would be predicted in each iteration
            TS_train_DF = pd.DataFrame(fcst_df_itr.iloc[:-1,:])
            TS_pred_DF = pd.DataFrame(fcst_df_itr.tail(1))
            
            # use scaler object to transform pred data for standardization
            ###
            #This part is throwing error
            if len(contdFS_new)>0:
                #chk1=3
                TS_pred_X1 = TSscaler.transform(TS_pred_DF.loc[:,contdFS_new])
                TS_pred_X1 = pd.DataFrame(TS_pred_X1 , columns = TS_pred_DF.loc[:,contdFS_new].columns)
                TS_pred_X = pd.merge(TS_pred_X1.reset_index(drop=True) , TS_pred_DF.loc[:,catgFS_new].reset_index(drop=True) , how='inner' , left_index = True , right_index = True)
            else:
                #chk1 = 4
                TS_pred_X=TS_pred_DF.loc[:,catgFS_new].reset_index(drop=True)
                
            
            if (modelNM == 'KNN'):                        
                # using KNN for time series forecast / prediction
                forecast_val = model_val.predict(TS_pred_X)[0][0]
                histDF_new.iloc[-1 , histDF_new.columns.get_loc('RTL_QTY')] = forecast_val
            
            elif (modelNM == 'LREG'):            
                # using Linear Regression for time series forecast / prediction
                forecast_val = model_val.predict(TS_pred_X)[0][0]
                
                histDF_new.iloc[-1 , histDF_new.columns.get_loc('RTL_QTY')] = forecast_val
            
            elif (modelNM == 'RFT'):
                
                forecast_val = model_val.predict(TS_pred_X)[0]
                
                histDF_new.iloc[-1 , histDF_new.columns.get_loc('RTL_QTY')] = forecast_val
                
            else:
                print('awaiting new forecast method')
                
            fcst_col1 = (modelNM + '_FCST')
            fcst_col2 = (modelNM + '_FLG')
                        
            TS_pred_DF = TS_pred_DF.loc[:,primary_key]
            TS_pred_DF[fcst_col1] = forecast_val
            TS_pred_DF[fcst_col2] = 1
            DF_pred_fcst = DF_pred_fcst.append(TS_pred_DF)
            #end_time_fcst_iter = time.time()
            
            
    return DF_pred_fcst,hads_prediction, param_val, cdf_all


"""
forecast engine
"""       

def clean_forecast(df):
    forecast_cols = ['KNN','LREG_NEW','RFT']
    df_cpy=df.copy()
    df.fillna(0,inplace=True)
    for i in forecast_cols:
        #df[i+'_FCST'] = df[i+'_FCST'].apply(lambda x: np.ceil(x))
        df[i+'_FLG'] = df[i+'_FCST'].apply(lambda x : 0 if x <= 0 else 1)
        df[i+'_FCST'] = df[i+'_FCST'].apply(lambda x: 0 if x < 0 else x)    
    return df


#fcst_hads, fcst_lads, primary_key, product_hrchy, binary_lag_var, continuous_lag_var, binary_no_lag_var, key_feature, params_feat_input = hads, lads, primary_key, product_hrchy, binary_lag_var, continuous_lag_var, binary_no_lag_var, key_feature, params_feat_input

def ML_forecast_engine(fcst_hads, fcst_lads, primary_key, product_hrchy, binary_lag_var, continuous_lag_var, binary_no_lag_var, key_feature, params_feat_input):
    ##global time_log
    #time_log = pd.DataFrame()
    terr_list = fcst_hads['territory'].unique()
    
    DF_opt_feature = pd.DataFrame()
    final_forecastDF = pd.DataFrame()
    overall_hads_jar = pd.DataFrame()
    error_terr_key = pd.DataFrame()
    Overall_op = pd.DataFrame()
    
    
    for terr in terr_list:
        hist_df_temp = fcst_hads[(fcst_hads['territory'] == terr)]
        var_list_temp = primary_key + product_hrchy + ['RTL_QTY']
        var_list_temp = list(set(var_list_temp))
        hist_df_temp = hist_df_temp.loc[:,var_list_temp]
        
        MonthSSNid = season_index_features(hist_df_temp)
        key_list = fcst_hads[(fcst_hads['territory'] == terr)]['KEY'].unique()
        
        for key in key_list:
            print(terr + " - " + key)
            hist_df = fcst_hads[(fcst_hads['territory'] == terr) & (fcst_hads['KEY'] == key)]
            
            try:
                lead_df = fcst_lads[(fcst_lads['territory'] == terr) & (fcst_lads['KEY'] == key)]
                if (len(hist_df.index) >= params_feat_input[3]) & (len(lead_df.index) == params_feat_input[4]) & (hist_df['RTL_QTY'].nunique() > 1):
                    model_run = 1
                    st_overall = time.time()
                    #logging.info(f'{terr} - {key} - has started')
                    var_list_temp = primary_key + product_hrchy + ['RTL_QTY'] + binary_lag_var + continuous_lag_var + binary_no_lag_var
                    var_list_temp = list(set(var_list_temp))
                    #len(var_list_temp)    
                    hist_df_temp = hist_df.loc[:,[i for i in var_list_temp if i in hist_df.columns]].reset_index()
                    #Editing the filteration
                    lead_df_temp = lead_df.loc[:, [i for i in var_list_temp if i in lead_df.columns]].reset_index()
                    hist_df_temp['ADS_FLG'] = 'H'
                    lead_df_temp['ADS_FLG'] = 'L'
                    df_temp = hist_df_temp.append(lead_df_temp).reset_index().sort_values(by=['TRDNG_WK_END_DT'])
                    #df_temp = df_temp.interpolate(method = 'polynomial',order=2,limit=60,axis=0,limit_direction = 'both')
                    df_temp = df_temp.fillna(0)
                    df_temp['TRDNG_WK_STRT_DT'] = pd.to_datetime(df_temp['TRDNG_WK_STRT_DT'],format='%d%b%Y')
                    df_temp['TRDNG_WK_END_DT'] = pd.to_datetime(df_temp['TRDNG_WK_END_DT'],format='%d%b%Y')
                    df_temp['YEAR_ID'] = df_temp['TRDNG_WK_STRT_DT'].dt.year
                    df_temp['MONTH_ID'] = df_temp['TRDNG_WK_STRT_DT'].dt.month
                    df_temp['MON_WK_ID'] = 1
                    df_temp['MON_WK_ID'] = df_temp.groupby(['territory','KEY','YEAR_ID','MONTH_ID'])['MON_WK_ID'].apply(lambda x: x.cumsum())
                    df_temp = pd.merge(df_temp, MonthSSNid, on=['GRP_NM','DPT_NM','CLSS_NM','SUB_CLSS_NM','MONTH_ID'], how='left')
                    st = time.time()
                    df_temp = days_since_last_event(df_temp, binary_lag_var,primary_key)        
                    df_temp = binary_lag_features(df_temp, binary_lag_var,primary_key)
                    df_temp = continuous_lag_features(df_temp, continuous_lag_var, params_feat_input[0],primary_key)
                    df_temp = fourier_term_features(df_temp, params_feat_input[1],primary_key)
                    
                    
                    hist_df = df_temp[(df_temp['ADS_FLG'] == 'H')].drop('ADS_FLG',1).sort_values(by=['TRDNG_WK_END_DT'])
                    lead_df = df_temp[(df_temp['ADS_FLG'] == 'L')].drop('ADS_FLG',1).sort_values(by=['TRDNG_WK_END_DT'])
                    lead_df['ROW_NUM'] = range(lead_df.shape[0])
                    # replace HADS & LADS numeric missing cases with 0
                    num_colNA = hist_df.select_dtypes(['int64','int32','float64','float32']).columns
                    hist_df[num_colNA] = hist_df[num_colNA].fillna(0 , inplace=False)
                    lead_df[num_colNA] = lead_df[num_colNA].fillna(0 , inplace=False)
                    # data preparation for forecast selection. sort HIST_DF
                    histFS_DF = hist_df.sort_values(by=['TRDNG_WK_END_DT'])
                    # create features using time series lag terms
                    histFS_DF = ts_lag_features(histFS_DF, params_feat_input[2],primary_key)
                    # create features using time series lag term stats (min / max / avg)
                    histFS_DF = lag_stat_features(histFS_DF, params_feat_input[2],primary_key)
                    # create bollinger band features
                    #histFS_DF = bng_band_features(histFS_DF, lag)
                    
                    
                    
                    # drop redundant auto python columns suchas level / index etc.
                    histFS_DF = histFS_DF.drop(['level_0','index'],1)
                    
                    # feature list without primary keys and other support columns
                    all_var_list = list(set(histFS_DF.columns.values) - set(primary_key) - set(product_hrchy) - set(['RTL_QTY']))
                            
                    # create feature set and target set for feature selection process
                    impFS_df = histFS_DF[0:len(hist_df.index)]
                    
                    # get feature type ie. categorical | continuous | constant
                    catgFS , contdFS , cnstFS = get_feature_type(impFS_df.loc[:,all_var_list])
                    
                    #do not eliminate records, but interpolate as this leads to missing out on important
                    #records
                    #impFS_Y = impFS_df.loc[-impFS_df.isnull().any(axis=1),['RTL_QTY']]
                    impFS_Y=impFS_df.fillna(method='backfill')['RTL_QTY']
                    #impFS_X = impFS_df.loc[-impFS_df.isnull().any(axis=1),all_var_list]
                    impFS_X=impFS_df.fillna(method='backfill')[all_var_list]
                    
                    # standardized DF (only continuous features) for feature selection process
                    FSscaler = StandardScaler()
                    
                    if len(contdFS)>0:
                        FSscaler.fit(impFS_X.loc[:,contdFS])
                        # transform feature set DF (only continuous features) based on scaler model object
                        impFS_XS = FSscaler.transform(impFS_X.loc[:,contdFS])
                        impFS_X1 = pd.DataFrame(impFS_XS , columns = impFS_X.loc[:,contdFS].columns)
                        # merge DF with categorical and continuous features
                        impFS_X = pd.merge(impFS_X1.reset_index(drop=True) , impFS_X.loc[:,catgFS].reset_index(drop=True) , how='inner' , left_index = True , right_index = True)
                    else:
                        impFS_X = impFS_X.loc[:,catgFS].reset_index(drop=True)
                    
                    impFS_Y = pd.DataFrame(impFS_Y).reset_index(drop=True)
                    
                    et = time.time()
                    logging.info(f'{terr} - {key} - Days since,  Lag Stats before feature selection - {et-st} seconds')
                    
                    # select features with high pearson correlation coeff
                    
                    st=time.time()
                    CORR_feature_list = CORR_feature_selection(impFS_X, impFS_Y, all_var_list)
                    et=time.time()
                    logging.info(f'{terr} - {key} - Correlation function - {et-st} seconds')
                    
                    # subset data for selected features in previous step
                    impFS_X = impFS_X.loc[:,CORR_feature_list]                
                    print(impFS_X.shape)
                    
                    # select features using Random Forest regression
                    
                    st = time.time()
                    RF_feature_list = RF_feature_selection(impFS_X, impFS_Y, CORR_feature_list)
                    et=time.time()
                    logging.info(f'{terr} - {key} - Random Forest feature selection - {et-st} seconds')
                    
                    # select features using Lasso regression
                    st = time.time()
                    LASSO_feature_list = LASSO_feature_selection(impFS_X, impFS_Y, RF_feature_list)
                    et = time.time()
                    logging.info(f'{terr} - {key} - LASSO feature selection - {et-st} seconds')
                    
                    # union selected features with manually defined key features
                    feature_list_final = list(LASSO_feature_list) + list(key_feature)
                    feature_list_final = list(set(feature_list_final))
                    
                    impFS_X = impFS_X.loc[:,[i for i in feature_list_final if i in impFS_X.columns]]
                    
                    print(impFS_X.shape)
                    # save selected features for each territory-key pair in a DF
                    DF_feature = pd.DataFrame(feature_list_final, columns=['KEY_FEATURE'])
                    DF_feature['territory'] = terr
                    DF_feature['KEY'] = key                        
                    
                    
                    
                    
                    # lead DF instance for current terriotry-key
                    TStemp_fcst = pd.DataFrame(lead_df.loc[:,primary_key])
                    
                    # using KNN for time series prediction (forecast)
                    #The below code outputs the lads prediction values, 
                    #we need to output the hads instances as well
                    
                    ############################RUNNING KNN MODEL#############################
                    st = time.time()
                    KNNpred_fcst,hads_prediction_knn, param_val_knn, Feat_imp_KNN = forecast_iteration(hist_df, lead_df, params_feat_input[4], feature_list_final, 'KNN',params_feat_input,primary_key)
                    et = time.time()
                    logging.info(f'{terr} - {key} - KNN Model Training and Prediction - {et-st} seconds')
                    
                    # join KNN forecast values with lead DF
                    TStemp_fcst = pd.merge(TStemp_fcst, KNNpred_fcst, on=primary_key, how='left')
                    
                    ###########################RUNNING LINEAR REGRESSION######################
                    # using Linear Regression (LREG) for time series prediction (forecast)
                    st = time.time()
                    LRpred_fcst,hads_prediction_lreg, param_val_lreg_new, Feat_imp_LREG  = forecast_iteration(hist_df, lead_df, params_feat_input[4], feature_list_final, 'LREG',params_feat_input,primary_key)
                    et = time.time()
                    logging.info(f'{terr} - {key} - LREG NEW Model Training and Prediction - {et-st} seconds')
                    
                    # join LREG forecast values with lead DF
                    TStemp_fcst = pd.merge(TStemp_fcst, LRpred_fcst, on=primary_key, how='left')
                    
                    ###########################RUNNING RANDOM FOREST MODEL####################
                    # using RF (Random Forest ) for time series prediction (forecast)                
                    
                    st = time.time()
                    RFTpred_fcst, hads_prediction_rf, param_val_rf, Feat_imp_RF = forecast_iteration(hist_df, lead_df, params_feat_input[4], feature_list_final, 'RFT',params_feat_input,primary_key)                
                    et = time.time()
                    logging.info(f'{terr} - {key} - RFT Model Training and Prediction - {et-st} seconds')
                    
                    # join LREG forecast values with lead DF
                    #Random Forest
                    TStemp_fcst = pd.merge(TStemp_fcst, RFTpred_fcst, on=primary_key, how='left')
                    
                    # append current output to final output DF
                    final_forecastDF = final_forecastDF.append(TStemp_fcst)
                    overall_hads=pd.merge(pd.merge(hads_prediction_knn,hads_prediction_lreg,on = ['territory', 'KEY', 'TRDNG_WK_STRT_DT', 'TRDNG_WK_END_DT'],how = 'left'),hads_prediction_rf,on = ['territory', 'KEY', 'TRDNG_WK_STRT_DT', 'TRDNG_WK_END_DT'],how = 'left')
                    overall_hads_jar=overall_hads_jar.append(overall_hads)
                    
                    ####ADD CODE OF ALL THE FEATURE IMPORTANCES HERE
                    
                    DF_feature_imps=pd.merge(pd.merge(DF_feature,
                             Feat_imp_LREG,how='outer',
                             left_on=['KEY_FEATURE'],right_on=['Columns']),Feat_imp_RF,left_on = ['KEY_FEATURE'],
                             right_on=['Features_RF'],how='outer')
                    
                    DF_feature_imps=DF_feature_imps.rename(columns = {'Columns':'Features_LREG'})
                    
                    DF_feature_imps['territory']=np.where(DF_feature_imps['territory'].isnull(),terr,DF_feature_imps['territory'])
                    DF_feature_imps['KEY']=np.where(DF_feature_imps['KEY'].isnull(),key,DF_feature_imps['KEY'])
                    
                    #####ADDING THE HYPERPARAMETERSS####
                    DF_feature_imps['KNN_Neighbours'] = param_val_knn
                    DF_feature_imps['RF_max_features'] = param_val_rf['max_features']
                    DF_feature_imps['RF_max_depth'] = param_val_rf['max_depth']
                    DF_feature_imps['RF_bootstrap'] = param_val_rf['bootstrap']
                                        
                    
                    
                    # append final shortlisted features in a DF
                    DF_opt_feature = DF_opt_feature.append(DF_feature_imps)
                    
                    final_forecastDF['TIME_FRAME'] = 'Lead Period'
                    overall_hads_jar['TIME_FRAME'] = 'Hist Period'
                    overall_hads_jar=overall_hads_jar.rename(columns={'HADS_Prediction_KNN':'KNN_FCST','HADS_Prediction_LREG':'LREG_FCST','HADS_Prediction_RF':'RFT_FCST'})
                    overall_hads_jar=overall_hads_jar.drop('RTL_QTY',axis=1)
                    Final_df  = final_forecastDF.append(overall_hads_jar)
                    
                    Final_df = Final_df.rename(columns = {'LREG_FCST':'LREG_NEW_FCST','LREG_FLG':'LREG_NEW_FLG'})
                    Final_df = clean_forecast(Final_df)
                    Final_df = Final_df.loc[:,['territory','KEY','TRDNG_WK_END_DT','TIME_FRAME','KNN_FCST','KNN_FLG','LREG_NEW_FCST','LREG_NEW_FLG','RFT_FCST','RFT_FLG']]
                    Overall_op = Overall_op.append(Final_df)
                    et_overall = time.time()
                    
                    logging.info(f'{terr} - {key} - Overall Time Taken - {et_overall-st_overall} seconds')
                                        
                else:
                    model_run=0
                    # lead DF instance for current terriotry-key
                    TStemp_fcst = pd.DataFrame(lead_df.loc[:,primary_key])
                    TStemp_fcst = TStemp_fcst.drop('TRDNG_WK_STRT_DT',axis=1)
                    if hist_df['RTL_QTY'].nunique()==1:
                        fcst_value = int(hist_df['RTL_QTY'].iloc[0])
                        if fcst_value == 0:
                            TStemp_fcst['KNN_FCST'] = fcst_value
                            TStemp_fcst['KNN_FLG'] = 0
                            TStemp_fcst['LREG_NEW_FCST'] = fcst_value
                            TStemp_fcst['LREG_NEW_FLG'] = 0
                            TStemp_fcst['RFT_FCST'] = fcst_value
                            TStemp_fcst['RFT_FLG'] = 0
                        else:
                            TStemp_fcst['KNN_FCST'] = fcst_value
                            TStemp_fcst['KNN_FLG'] = 1
                            TStemp_fcst['LREG_NEW_FCST'] = fcst_value
                            TStemp_fcst['LREG_NEW_FLG'] = 1
                            TStemp_fcst['RFT_FCST'] = fcst_value
                            TStemp_fcst['RFT_FLG'] = 1
                    else:
                        # assign 0 as forecast value for KNN | LREG method
                        TStemp_fcst['KNN_FCST'] = 0
                        TStemp_fcst['KNN_FLG'] = 0
                        TStemp_fcst['LREG_NEW_FCST'] = 0
                        TStemp_fcst['LREG_NEW_FLG'] = 0
                        TStemp_fcst['RFT_FCST'] = 0
                        TStemp_fcst['RFT_FLG'] = 0
                    
                    
                    overall_hads_jar = hist_df.loc[:,['territory','KEY','TRDNG_WK_END_DT']].copy()
                    overall_hads_jar['KNN_FCST'] = hist_df['RTL_QTY']
                    overall_hads_jar['KNN_FLG'] = 0
                    overall_hads_jar['LREG_NEW_FCST'] = hist_df['RTL_QTY']
                    overall_hads_jar['LREG_NEW_FLG'] = 0
                    overall_hads_jar['RFT_FCST'] = hist_df['RTL_QTY']
                    overall_hads_jar['RFT_FLG'] = 0
                    overall_hads_jar['TIME_FRAME'] = 'Hist Period'
                    
                    # append current output to final output DF
                    final_forecastDF = final_forecastDF.append(TStemp_fcst)
                    final_forecastDF['TIME_FRAME']='Lead Period'
                    Final_df  = final_forecastDF.append(overall_hads_jar)
                    Final_df = clean_forecast(Final_df)
                    Final_df = Final_df.loc[:,['territory','KEY','TRDNG_WK_END_DT',
                                               'TIME_FRAME','KNN_FCST','KNN_FLG','LREG_NEW_FCST','LREG_NEW_FLG','RFT_FCST','RFT_FLG']]
                    
                    Overall_op = Overall_op.append(Final_df)
            except:
                model_run=-1
                Final_df = pd.DataFrame({'territory':[0],
                             'KEY':[0],
                             'TRDNG_WK_END_DT':[0],
                             'TIME_FRAME':[0],
                             'KNN_FCST':[0],
                             'KNN_FLG':[0],
                             'LREG_NEW_FCST':[0],
                             'LREG_NEW_FLG':[0],
                             'RFT_FCST':[0],
                             'RFT_FLG':[0]})
                Final_df=Final_df.drop(0)
                Overall_op = Overall_op.append(Final_df)
                #logging.error('Error occurred for '+ str(terr) + '-'+str(key))
                logging.info(f'{terr} - {key} - Error - Zero')
                pass
                continue
            
    return Overall_op,DF_opt_feature




#run_models(dataframes_hads,dataframes_lads, primary_key,product_hrchy,target_var,binary_lag_var, continuous_lag_var, binary_no_lag_var, key_feature, params_feat_input)
def run_models(dataframes_hads, dataframes_lads, primary_key, product_hrchy, binary_lag_var, continuous_lag_var, binary_no_lag_var, key_feature,params_feat_input):
    os.environ['NUMEXPR_MAX_THREADS'] = '24'    
    with concurrent.futures.ProcessPoolExecutor(max_workers= 32) as executor:
        #overall_hads_jar,error_terr_key, DF_opt_feature  
        op = list(tqdm.tqdm(executor.map(ML_forecast_engine, dataframes_hads, dataframes_lads, itertools.repeat(primary_key), itertools.repeat(product_hrchy), itertools.repeat(binary_lag_var), itertools.repeat(continuous_lag_var), itertools.repeat(binary_no_lag_var),itertools.repeat(key_feature),itertools.repeat(params_feat_input)),  total = len(dataframes_hads)))
        #, overall_hads_jar,error_terr_key, DF_opt_feature
        return op


