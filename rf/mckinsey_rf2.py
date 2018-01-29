import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score, cross_val_predict

DATAPATH = 'F:/srikanth/data/k_data/AV/Mckinsay_Hackathon'

train = pd.read_csv(os.path.join(DATAPATH,'train.csv'))
test = pd.read_csv(os.path.join(DATAPATH,'test.csv'))

bank_acc = train.Customer_Existing_Primary_Bank_Code.value_counts(dropna=False)
bank_acc_rare = list(bank_acc[bank_acc<40].index)
train.ix[train['Customer_Existing_Primary_Bank_Code'].isin(bank_acc_rare), "Customer_Existing_Primary_Bank_Code"] = "Others"

bank_source = train.Source.value_counts(dropna=False)
bank_src_rare = list(bank_source[bank_source<100].index)
train.ix[train['Source'].isin(bank_src_rare), "Source"] = "Others"

c_code = train.City_Code.value_counts(dropna=False)
city_rare = list(c_code[c_code < 100].index)
train.ix[train['City_Code'].isin(city_rare), 'City_Code'] = "Others"

emp_code = train.Employer_Code.value_counts(dropna=False)
emp_code_rare = list(emp_code[emp_code<40].index)
train.ix[train['Employer_Code'].isin(emp_code_rare), "Employer_Code"] = "Others"

# # Preprocessing
train2 = train.copy()
id_train = train['ID']
label = train['Approved']

dropCols = ['ID', 'Approved', 'DOB', 'Lead_Creation_Date']
train2.drop(dropCols, axis=1, inplace = True)

y_train = label
X_train = pd.get_dummies(train2)
print('Train complete shape after dummy pandas preprocessing :-', X_train.shape)

test.ix[test['Customer_Existing_Primary_Bank_Code'].isin(bank_acc_rare), "Customer_Existing_Primary_Bank_Code"] = "Others"

test.ix[test['Source'].isin(bank_src_rare), "Source"] = "Others"

test.ix[test['City_Code'].isin(city_rare), 'City_Code'] = "Others"
newcities = list(set(test['City_Code']) - set(train['City_Code']))
test.ix[test['City_Code'].isin(newcities), 'City_Code'] = np.nan

test.ix[test['Employer_Code'].isin(emp_code_rare), "Employer_Code"] = "Others"
newempnames = list(set(test['Employer_Code']) - set(train['Employer_Code']))
test.ix[test['Employer_Code'].isin(newempnames), "Employer_Code"] = "Others"

testdropcols = list(set(dropCols)-set(['Approved']))
test2 = test.drop(testdropcols, axis=1)

X_test = pd.get_dummies(test2)
missingCols = list(set(X_train.columns)-set(X_test.columns))
for col in missingCols:
    X_test[col] = 0
X_test = X_test[X_train.columns]
assert X_train.columns.equals(X_test.columns)

 # Modeling
X_train_2 = X_train.fillna(-999)
X_test_2 = X_test.fillna(-999)

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
    clf.fit(X_train_2, y_train)
    test_preds = clf.predict_proba(X_test_2)[:,1]
    print("RF %s done" % i)

    submission = pd.DataFrame({'ID':test['ID'], 'Approved':test_preds})
    submission = submission[['ID', 'Approved']]
    submission.to_csv(os.path.join(DATAPATH,"results/random_forest","Sub%s.csv") % str(numbers[i]), index = False)
