import numpy as np
import pandas as pd
import xgboost as xgb
import os
DATAPATH = 'F:/srikanth/data/k_data/AV/Mckinsay_Hackathon'

train = pd.read_csv(os.path.join(DATAPATH,"train_preprocessed.csv"))
test = pd.read_csv(os.path.join(DATAPATH,"test_preprocessed.csv"))
labels = pd.read_csv(os.path.join(DATAPATH,"train_labels.csv"), header = None)
test_ids = pd.read_csv(os.path.join(DATAPATH,"test_ids.csv"), header = None)

labels = list(labels.iloc[:,0])
test_ids = list(test_ids.iloc[:,0])

params = {'booster':'gbtree', 
          'objective':'binary:logistic', 
          'max_depth':9, 
          'eval_metric':'auc',
          'eta':0.02, 
          'silent':1, 
          'nthread':14, 
          'subsample': 0.9, 
          'colsample_bytree':0.9, 
          'scale_pos_weight': 1,
          'min_child_weight':3, 
          'max_delta_step':3}

num_rounds = 400

params['seed'] = 12345678 # 0.85533
dtrain = xgb.DMatrix(train, labels, missing=np.nan)
xgb.cv(params, dtrain, num_rounds, nfold=4)
clf = xgb.train(params, dtrain, num_rounds)
dtest = xgb.DMatrix(test, missing = np.nan)
test_preds = clf.predict(dtest)

submission = pd.DataFrame({ 'ID':test_ids, 'Approved':test_preds})
submission = submission[['ID', 'Approved']]
submission.to_csv(os.path.join(DATAPATH,"results","xgb2_final.csv"), index = False)
