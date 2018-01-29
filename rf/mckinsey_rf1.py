import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score, cross_val_predict

DATAPATH = 'F:/srikanth/data/k_data/AV/Mckinsay_Hackathon'

train = pd.read_csv(os.path.join(DATAPATH,"train_preprocessed.csv"))
test = pd.read_csv(os.path.join(DATAPATH,"test_preprocessed.csv"))
labels = pd.read_csv(os.path.join(DATAPATH,"train_labels.csv"), header = None)
test_ids = pd.read_csv(os.path.join(DATAPATH,"test_ids.csv"), header = None)

labels = list(labels.iloc[:,0])
test_ids = list(test_ids.iloc[:,0])

# # Modeling
X_train_2 = train.fillna(-999)
X_test_2 = test.fillna(-999)

# from sklearn.cross_validation import KFold
# kf = KFold(len(X_train_2), n_folds=4)
# scores = cross_val_score(clf, X_train_2, y_train, scoring='roc_auc', cv=kf)
# print "CV:", np.mean(scores), "+/-", np.std(scores), "All:", scores
# CV: 0.831889207925 +/- 0.0109754348042 All: [ 0.82381549  0.82907869  0.85055107  0.82411158]
seeds = [31121421,53153,5245326,6536,75]
numbers = [151,152,153,154,155]

for i in range(5): 
    clf = RandomForestClassifier(n_estimators=360, max_depth=9, criterion = 'entropy', min_samples_split=2,
                                 bootstrap = False, n_jobs=-1, random_state=seeds[i])
    clf.fit(X_train_2,labels)
    test_preds = clf.predict_proba(X_test_2)[:,1]
    print("RF %s done" % i)

    submission = pd.DataFrame({'ID':test_ids, 'Approved':test_preds})
    submission = submission[['ID', 'Approved']]
    submission.to_csv(os.path.join(DATAPATH,'results/random_forest',"rf%s.csv") % str(numbers[i]), index = False)
