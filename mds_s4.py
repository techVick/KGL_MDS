"""TechVick_GonnaWin"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


testKag=pd.read_csv("D:/Vikulp_imp_ambitions/machine_learning/kaggle/mds/test.csv")
train=pd.read_csv("D:/Vikulp_imp_ambitions/machine_learning/kaggle/mds/train.csv")
sample_submission=pd.read_csv("D:/Vikulp_imp_ambitions/machine_learning/kaggle/mds/sample_submission.csv")


train = pd.get_dummies(train, columns=['X0','X1','X2','X3','X4','X5','X6','X8'])
"""TEST TRAIN SPLIT / TEST SCORE"""
#trainR, testR = train_test_split(train, test_size = 0.2)
#
#target1 = trainR["y"].values
#trainR.pop('y')
#trainR.pop('ID')
#feature1 = trainR
#target2 = testR["y"].values
#testR.pop('y')
#testR.pop('ID')
#feature2 = testR
#regr_1=RandomForestRegressor(max_depth=3)
#regr_1.fit(feature1,target1)
#y_1=regr_1.predict(feature2)
#s1=r2_score(target2,y_1)
#print(s1)
""" this is to test best tree depth"""
#for i in range(1,50):
#    regr_1=RandomForestRegressor(max_depth=i)
#    regr_1.fit(feature1,target1)
#    y_1=regr_1.predict(feature2)
#    s1=r2_score(target2,y_1)
#    print(s1)
"""END"""
testKag = pd.get_dummies(testKag, columns=['X0','X1','X2','X3','X4','X5','X6','X8'])
# get the columns in train that are not in test
con=pd.concat((train, testKag))
col_to_add = np.setdiff1d(con.columns, train.columns)
#print(testKag.dtypes)
# add these columns to test, setting them equal to zero
for c in col_to_add:
    train[c] = 0
col_to_add = np.setdiff1d(con.columns, testKag.columns)
#print(testKag.dtypes)
# add these columns to test, setting them equal to zero
for c in col_to_add:
    testKag[c] = 0

target1 = train["y"].values
train.pop('y')
train.pop('ID')
feature1 = train

regr_1=RandomForestRegressor(max_depth=3)
regr_1.fit(feature1,target1)



#print(train)
id=testKag['ID']
testKag.pop('ID')
testKag.pop('y')
feature2 = testKag
y_1=regr_1.predict(feature2)
data_frame = pd.DataFrame(id,columns=['ID'])
data_frame['y'] = pd.Series(y_1, index=data_frame.index)
data_frame['ID'] = data_frame["ID"].astype(int)
np.savetxt('submission3.csv',data_frame,delimiter=',')






