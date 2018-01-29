import pandas as pd
import numpy as np
import os
import datetime

FILEPATH = 'F:/srikanth/data/k_data/AV/Mckinsay_Hackathon'

train = pd.read_csv(os.path.join(FILEPATH,'train.csv'))
test = pd.read_csv(os.path.join(FILEPATH,'test.csv'))

DOB=[]
Lead_Creation_Date=[]
#train['DOB'].fillna(0, inplace=True)
#train['Lead_Creation_Date'].fillna('07/09/16', inplace=True)

for dd in train.DOB:
    dt_split = dd.split('/')
    if len(dt_split[2])==2:
        daa = dt_split[0]+'/'+dt_split[1]+'/'+'19'
        DOB.append(daa+dt_split[2])
    else:
        DOB.append(dd)
df = pd.DataFrame({'DOB':DOB})

for dd in train.Lead_Creation_Date:
    dt_split = dd.split('/')
    if len(dt_split[2])==2:
        daa = dt_split[0]+'/'+dt_split[1]+'/'+'19'
        Lead_Creation_Date.append(daa+dt_split[2])
    else:
        Lead_Creation_Date.append(dd)
df['Lead_Creation_Date'] = pd.Series(Lead_Creation_Date)

noofDays = []
for i in range(len(train.DOB)):
    noofDays.append((pd.to_datetime(train['Lead_Creation_Date'][i]) - pd.to_datetime(train['DOB'][i])).days)
df['noofDays'] = pd.Series(noofDays)

train['DOB'] = df['DOB']
train['Lead_Creation_Date'] = df['Lead_Creation_Date']
train['noofDays'] = df['noofDays']

train['dob_day'] = pd.to_datetime(train['DOB']).dt.day
train['dob_dayofweek'] = pd.to_datetime(train['DOB']).dt.dayofweek
train['dob_weekofyear'] = pd.to_datetime(train['DOB']).dt.weekofyear
train['dob_quarter'] = pd.to_datetime(train['DOB']).dt.quarter
train['dob_month'] = pd.to_datetime(train['DOB']).dt.month
train['dob_year'] = pd.to_datetime(train['DOB']).dt.year

train['lcd_day'] = pd.to_datetime(train['Lead_Creation_Date']).dt.day
train['lcd_dayofweek'] = pd.to_datetime(train['Lead_Creation_Date']).dt.dayofweek
train['lcd_weekofyear'] = pd.to_datetime(train['Lead_Creation_Date']).dt.weekofyear
train['lcd_quarter'] = pd.to_datetime(train['Lead_Creation_Date']).dt.quarter
train['lcd_month'] = pd.to_datetime(train['Lead_Creation_Date']).dt.month
train['lcd_year'] = pd.to_datetime(train['Lead_Creation_Date']).dt.year

#moving accounts, city codes to 'other' category, whose bank has less than 40 accounts
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


train['City_Category'] = train['City_Category'].astype(str)
train['City_Code'] = train['City_Code'].astype(str)
train['Employer_Category1'] = train['Employer_Category1'].astype(str)
train['Employer_Category2'] = train['Employer_Category2'].astype(str)
train['Primary_Bank_Type'] = train['Primary_Bank_Type'].astype(str)
train['Source_Category'] = train['Source_Category'].astype(str)


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train['City_Category_encoded'] = le.fit_transform(train['City_Category'])
train['City_Code_encoded'] = le.fit_transform(train['City_Code'])
train['Employer_Category1_encoded'] = le.fit_transform(train['Employer_Category1'])
train['Employer_Category2_encoded'] = le.fit_transform(train['Employer_Category2'])
train['Primary_Bank_Type_encoded'] = le.fit_transform(train['Primary_Bank_Type'])
train['Source_Category_encoded'] = le.fit_transform(train['Source_Category'])

train2 = train.copy()
id_train = train['ID']
label = train['Approved']

dropCols = ['ID', 'Approved', 'DOB', 'Lead_Creation_Date']
train2.drop(dropCols, axis=1, inplace = True)

y_train = label
X_train = pd.get_dummies(train2)
print('Train complete shape after dummy pandas preprocessing :-', X_train.shape)

#for test processing
DOB= []
Lead_Creation_Date = []

test['DOB'].fillna('09/11/1989', inplace=True)
test['Lead_Creation_Date'].fillna('07/09/2016', inplace=True)

for dd in test.DOB:
    dt_split = dd.split('/')
    if len(dt_split[2])==2:
        daa = dt_split[0]+'/'+dt_split[1]+'/'+'19'
        DOB.append(daa+dt_split[2])
    else:
        DOB.append(dd)
df = pd.DataFrame({'DOB':DOB})

for dd in test.Lead_Creation_Date:
    dt_split = dd.split('/')
    if len(dt_split[2])==2:
        daa = dt_split[0]+'/'+dt_split[1]+'/'+'19'
        Lead_Creation_Date.append(daa+dt_split[2])
    else:
        Lead_Creation_Date.append(dd)
df['Lead_Creation_Date'] = pd.Series(Lead_Creation_Date)

noofDays = []
counter = 0
for i in range(len(test.DOB)):
    noofDays.append((pd.to_datetime(test['Lead_Creation_Date'][i]) - pd.to_datetime(test['DOB'][i])).days)
df['noofDays'] = pd.Series(noofDays)

test['DOB'] = df['DOB']
test['Lead_Creation_Date'] = df['Lead_Creation_Date']
test['noofDays'] = df['noofDays']


test['lcd_day'] = pd.to_datetime(test['Lead_Creation_Date']).dt.day
test['lcd_dayofweek'] = pd.to_datetime(test['Lead_Creation_Date']).dt.dayofweek
test['lcd_weekofyear'] = pd.to_datetime(test['Lead_Creation_Date']).dt.weekofyear
test['lcd_quarter'] = pd.to_datetime(test['Lead_Creation_Date']).dt.quarter
test['lcd_month'] = pd.to_datetime(test['Lead_Creation_Date']).dt.month
test['lcd_year'] = pd.to_datetime(test['Lead_Creation_Date']).dt.year

test['dob_day'] = pd.to_datetime(test['DOB']).dt.day
test['dob_dayofweek'] = pd.to_datetime(test['DOB']).dt.dayofweek
test['dob_weekofyear'] = pd.to_datetime(test['DOB']).dt.weekofyear
test['dob_quarter'] = pd.to_datetime(test['DOB']).dt.quarter
test['dob_month'] = pd.to_datetime(test['DOB']).dt.month
test['dob_year'] = pd.to_datetime(test['DOB']).dt.year

#moving accounts, city codes to 'other' category, whose bank has less than 40 accounts
test.ix[test['Customer_Existing_Primary_Bank_Code'].isin(bank_acc_rare), "Customer_Existing_Primary_Bank_Code"] = "Others"

test.ix[test['Source'].isin(bank_src_rare), "Source"] = "Others"

test.ix[test['City_Code'].isin(city_rare), 'City_Code'] = "Others"
newcities = list(set(test['City_Code']) - set(train['City_Code']))
test.ix[test['City_Code'].isin(newcities), 'City_Code'] = np.nan

test.ix[test['Employer_Code'].isin(emp_code_rare), "Employer_Code"] = "Others"
newempnames = list(set(test['Employer_Code']) - set(train['Employer_Code']))
test.ix[test['Employer_Code'].isin(newempnames), "Employer_Code"] = "Others"

test['City_Category'] = test['City_Category'].astype(str)
test['City_Code'] = test['City_Code'].astype(str)
test['Employer_Category1'] = test['Employer_Category1'].astype(str)
test['Employer_Category2'] = test['Employer_Category2'].astype(str)
test['Primary_Bank_Type'] = test['Primary_Bank_Type'].astype(str)
test['Source_Category'] = test['Source_Category'].astype(str)

#one hot encoding -- i think these should not done for test set, we'll see
#test['City_Code_encoded'] = le.transform(test['City_Code'])
test['City_Category_encoded'] = le.fit_transform(test['City_Category'])
test['Employer_Category1_encoded'] = le.fit_transform(test['Employer_Category1'])
test['Employer_Category2_encoded'] = le.fit_transform(test['Employer_Category2'])
test['Primary_Bank_Type_encoded'] = le.fit_transform(test['Primary_Bank_Type'])
test['Source_Category_encoded'] = le.fit_transform(test['Source_Category'])

testdropcols = list(set(dropCols)-set(['Approved']))
test2 = test.drop(testdropcols, axis=1)

X_test = pd.get_dummies(test2)
missingCols = list(set(X_train.columns)-set(X_test.columns))
for col in missingCols:
    X_test[col] = 0
X_test = X_test[X_train.columns]
assert X_train.columns.equals(X_test.columns)

X_train.to_csv(os.path.join(FILEPATH,"train_preprocessed.csv"), index = False)
X_test.to_csv(os.path.join(FILEPATH,"test_preprocessed.csv"), index = False)
y_train.to_csv(os.path.join(FILEPATH,"train_labels.csv"), index = False)
test['ID'].to_csv(os.path.join(FILEPATH,"test_ids.csv"), index = False)