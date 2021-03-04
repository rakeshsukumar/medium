# -*- coding: utf-8 -*-
"""
"""

from fbprophet import Prophet
import time
import pandas as pd
import numpy as np
import tqdm
import warnings
from itertools import repeat
import multiprocessing as mp
import time
import tqdm
import os
import concurrent.futures
import logging
import multiprocessing as mp
import datetime
warnings.filterwarnings('ignore')
from sklearn.model_selection import ParameterGrid
warnings.filterwarnings('ignore')
import dask
from distributed import Client, performance_report
import fbprophet.diagnostics
import itertools
import functools
import dask.distributed
from fbprophet.diagnostics import cross_validation, performance_metrics


#******************************************Logging Configuration********************************************************
def logging_(path):   
    logging.getLogger('fbprophet').setLevel(logging.WARNING)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, filename= path, filemode = 'w', format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    
    
    
# # Data Import and renaming columns




#****************************************************Training(HADS) Data**********************************************
def HADS_data(path):
    df = pd.read_csv(path, delimiter= ',')
    df['TRDNG_WK_END_DT'] = pd.to_datetime(df['TRDNG_WK_END_DT'], infer_datetime_format = True)
    df = df.rename(columns={'TRDNG_WK_END_DT': 'ds', 'RTL_QTY': 'y'})
    gb = df.groupby(['territory', 'KEY'])
    dataframes = [gb.get_group(x) for x in gb.groups]
    return df , dataframes


#*********************************************************LADS data******************************************************8

def LADS_data(path):
    df_test = pd.read_csv(path, delimiter= ',')
    df_test['TRDNG_WK_END_DT'] = pd.to_datetime(df_test['TRDNG_WK_END_DT'], infer_datetime_format = True)
    df_test = df_test.rename(columns={'TRDNG_WK_END_DT': 'ds'})
    return df_test



def single_param_cv(history_df, metrics, param_dict, cols, peak_var):
    m = fbprophet.Prophet(**param_dict)
    for c in cols:
        if c in (peak_var):
            m.add_regressor(c, mode = 'multiplicative')
        else:
            m.add_regressor(c, prior_scale=0.01)
    m.fit(history_df)
    
    if len(history_df)>= 52 and  len(history_df) < 104:
        
        df_cv = cross_validation(m, initial='210 days', period='28 days', horizon = '56 days')
    elif len(history_df)>= 104:
        df_cv = cross_validation(m, initial='364 days', period='70 days', horizon = '140 days')
    df_p = performance_metrics(df_cv, rolling_window=1)
    df_p['params'] = str(param_dict)
    df_p = df_p.loc[:, metrics]
    return df_p


def param_grid_to_df(**param_dict):
    param_iter = itertools.product(*param_dict.values())
    params =[]
    for param in param_iter:
        params.append(param) 
    params_df = pd.DataFrame(params, columns=list(param_dict.keys()))
    return params_df


def hyperparameter_cv(history_df, params_df, single_cv_callable, pool):
    results = []
    for param in params_df.values:
        param_dict = dict(zip(params_df.keys(), param))
        if pool is None:
            predict = single_cv_callable(history_df, param_dict=param_dict)
            results.append(predict)
        elif isinstance(pool, dask.distributed.client.Client):
            remote_df = pool.scatter(history_df)
            future = pool.submit(single_cv_callable, remote_df, param_dict=param_dict)
            results.append(future)
        else:
            logging.error('Pool needs to be an instantiated dask distributed client object or None')
    if isinstance(pool, dask.distributed.client.Client):
        results = pool.gather(results)
    results_df = pd.concat(results)
    
    return results_df

def param_tuning(train, client, cols, peak_var):
    try:
        pool = client
        train = train.sort_values(by=['ds'])
        if len(train)>= 52 and  len(train) < 104:
            param_dict = {'changepoint_prior_scale': [0.001, 0.01, 0.1,0.2,0.3,0.4, 0.5, 0.6, 0.7, 0.8,0.9,1],
                       'changepoint_range': [0.8],
                 'yearly_seasonality' : [False],

              }
            metrics = ['horizon', 'rmse', 'mape', 'params'] 

            params_df = param_grid_to_df(**param_dict)
            single_cv_callable = functools.partial(single_param_cv, metrics=metrics, cols=cols, peak_var=peak_var)
            model_parameters  = hyperparameter_cv(train, params_df, single_cv_callable, pool)
            parameters = model_parameters.sort_values(by=['rmse'], ascending = True)
            parameters = parameters.reset_index(drop=True)
            parameters = parameters.head(1)

        elif len(train) >= 104:
            param_dict = {'changepoint_prior_scale': [0.001, 0.01, 0.1,0.2,0.3,0.4, 0.5, 0.6, 0.7, 0.8,0.9,1],
                           'changepoint_range': [0.8],
                     'yearly_seasonality' : [10],'seasonality_mode':['multiplicative','additive']

                  }

            metrics = ['horizon', 'rmse', 'mape', 'params'] 

            params_df = param_grid_to_df(**param_dict)
            single_cv_callable = functools.partial(single_param_cv, metrics=metrics, cols=cols, peak_var=peak_var)
            model_parameters  = hyperparameter_cv(train, params_df, single_cv_callable, pool)
            parameters = model_parameters.sort_values(by=['rmse'], ascending = True)
            parameters = parameters.reset_index(drop=True)
            parameters = parameters.head(1)

        elif len(train) < 52:


            params = 0
            horizon = 0
            parameters = pd.DataFrame(columns = ['horizon','rmse','mape', 'params'])
            parameters = parameters.append({'horizon':horizon, 'params':params},ignore_index=True)

        parameters['territory'] = train['territory'].iloc[0]
        parameters['KEY'] = train['KEY'].iloc[0]

        return parameters
    except:
        logging.error('error occcured for ' + str(train['territory'].iloc[0]) + str(' ') + str(train['KEY'].iloc[0]))
        pass 
    


def forecast_(train, test, p, cols, peak_var):
    if len(train) < 52:
        try:
            train = train.sort_values(by=['ds'])
            test = test.sort_values(by=['ds'])
            full_forecast = pd.concat([train, test])
            full_forecast = full_forecast[['ds', 'y' ,'KEY' , 'territory', 'TRDNG_WK_STRT_DT', 'GRP_NM', 'DPT_NM', 'CLSS_NM', 'SUB_CLSS_NM']]
            full_forecast['yhat'] = None
            full_forecast['PRO_FLG'] =0
            
            return full_forecast
         
        except:
            logging.error('error occcured for ' + str(test['territory'].iloc[0]) + str(' ') + str(test['KEY'].iloc[0]))
            pass 
    else:
        try:
            
            train = train.sort_values(by=['ds'])
            test = test.sort_values(by=['ds'])
            if len(train) >= 52 and  len(train) < 104:
                m=Prophet(changepoint_prior_scale = p['changepoint_prior_scale'],
                              changepoint_range = p['change_point_range'],
                               yearly_seasonality = False,
                               interval_width=0.95)
        
            elif len(train) >= 104:
                m=Prophet(changepoint_prior_scale = p['changepoint_prior_scale'],
                          changepoint_range = p['change_point_range'],seasonality_mode = p['seasonality_mode'],
                           yearly_seasonality = 10,
                           interval_width=0.95)
            
                
                
    
    
    
            for c in cols:
                    if c in (peak_var):
                   
                        m.add_regressor(c, mode = 'multiplicative')
                    else:
                        m.add_regressor(c, prior_scale=0.01)
            
    
    #********************************************************** Train the model**********************************************
            m.fit(train)
            
           
    #********************************************************** Make Predictions********************************************
            future_pd = m.make_future_dataframe(
            periods=len(test), 
            freq='W-SAT', 
            include_history=True
            )
            d_ = pd.concat([train, test])
            future_pd = pd.merge(future_pd, d_,  on ='ds')
            forecast_pd = m.predict( future_pd )
            final = forecast_pd[['ds', 'yhat' ]]
            full_forecast = pd.merge(final,d_[['ds', 'y' ,'KEY' , 'territory', 'TRDNG_WK_STRT_DT', 'GRP_NM', 'DPT_NM', 'CLSS_NM', 'SUB_CLSS_NM']] )
            full_forecast['TRDNG_WK_STRT_DT'] = pd.to_datetime(full_forecast['TRDNG_WK_STRT_DT'], infer_datetime_format = True)
            full_forecast['PRO_FLG'] = 1
            
    #*************************************Flagging negative values as zero*******************************************************************
    
            t_ = full_forecast[full_forecast['ds']>=test['ds'].min()]
            if (any(t_['yhat']<0)==True):
                full_forecast['PRO_FLG'] = 0
            return full_forecast
        
        except:
            logging.error('error occcured for ' + str(test['territory'].iloc[0]) + str(' ') + str(test['KEY'].iloc[0]))
            pass 
        

def run_fore(iterable):
    
    logging.info('Forecast started')
    pool = mp.Pool(mp.cpu_count())
    start_time = time.time()
    os.environ['NUMEXPR_MAX_THREADS'] = '24'
        
    fore_all = pool.starmap(forecast_, iterable)
    pool.close()
    pool.join()
    
    time_ = time.time()- start_time
    logging.info('Forecast ended with time taken  ' + str(time_))
    return fore_all
