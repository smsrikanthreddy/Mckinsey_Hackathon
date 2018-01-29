import pandas as pd
import numpy as np
from scipy.stats import rankdata
import os

DATAPATH = 'F:/srikanth/data/k_data/AV/Mckinsay_Hackathon'

xgb_pred = pd.read_csv(os.path.join(DATAPATH,"results","XGB_Ens.csv")) #XGB
ftrl_pred = pd.read_csv(os.path.join(DATAPATH,"results","ftrl_final.csv")) #FTRL
xgb_ftrl_pred = pd.read_csv(os.path.join(DATAPATH,"results","xgboost_ftrl_results.csv"))
rf_pred = pd.read_csv(os.path.join(DATAPATH,"results","RF_Ens.csv")) #RF

ens = xgb_pred.copy()
ens.rename(columns={'Approved':'XGB'}, inplace = True)
ens['RF'] = rf_pred['Approved']
ens['FTRL'] = ftrl_pred['Approved']
ens['XGB_FTRL'] = ftrl_pred['Approved']

ens['XGB1_Rank'] = rankdata(ens['XGB'], method='min')
ens['RF_Rank'] = rankdata(ens['RF'], method='min')
ens['FTRL_Rank'] = rankdata(ens['FTRL'], method='min')
ens['XGB_FTRL_Rank'] = rankdata(ens['XGB_FTRL'], method='min')

ens['Final'] = (0.75*ens['XGB'] + 0.25*ens['RF']) * 0.75 + 0.25 * (0.5* ens['FTRL']+0.5*ens['XGB_FTRL'])

ens = ens[['ID', 'Final']]
ens.rename(columns={'Final':'Approved'}, inplace = True)
ens.sort_index(inplace = True)
ens.head()

ens.to_csv(os.path.join(DATAPATH,"results","FinalSolution.csv"), index = False)