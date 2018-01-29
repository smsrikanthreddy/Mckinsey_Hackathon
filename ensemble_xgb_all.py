import pandas as pd
import numpy as np
from scipy.stats import rankdata
import os

DATAPATH = 'F:/srikanth/data/k_data/AV/Mckinsay_Hackathon'

test = pd.read_csv(os.path.join(DATAPATH,"test.csv"), usecols = ["ID"])

preds = pd.read_csv(os.path.join(DATAPATH,"results","xgb_final.csv"))
preds['ID'] = test['ID']
preds1 = pd.read_csv(os.path.join(DATAPATH,"results","xgb1_final.csv"))
preds2 = pd.read_csv(os.path.join(DATAPATH,"results","xgb2_final.csv"))
preds3 = pd.read_csv(os.path.join(DATAPATH,"results","xgb3_final.csv"))

preds['pred1_RANK'] = rankdata(preds['Approved'], method='ordinal')
preds1['pred2_RANK'] = rankdata(preds1['Approved'], method='ordinal')
preds2['pred3_RANK'] = rankdata(preds2['Approved'], method='ordinal')
preds3['pred4_RANK'] = rankdata(preds3['Approved'], method='ordinal')

preds['Approved'] = 0.2 * (preds['Approved'] +  preds1['Approved'] +preds2['Approved'] + preds3['Approved'])

preds.to_csv(os.path.join(DATAPATH,"results","XGB_Ens.csv"), index = False)