"""TechVick_GonnaWin"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import GradientBoostingRegressor


testKag=pd.read_csv("D:/Vikulp_imp_ambitions/machine_learning/kaggle/mds/test.csv")
train=pd.read_csv("D:/Vikulp_imp_ambitions/machine_learning/kaggle/mds/train.csv")
sample_submission=pd.read_csv("D:/Vikulp_imp_ambitions/machine_learning/kaggle/mds/sample_submission.csv")

column=['X0','X1','X2','X3','X4','X5','X6','X8']
for i in column:
    input = train[[i]]
    output = []
    for k in input[i]:
        num=sum([ord(j) for j in k],0)
        number = num-96
        output.append(number)
    output=[int(m) for m in output]
    df = pd.DataFrame(np.array(output).reshape(4209,1), columns = list("a"))
    train[i]=df['a']


"""TEST TRAIN SPLIT / TEST SCORE"""
#trainR, testR = train_test_split(train, test_size = 0.1)
#
#target1 = trainR["y"].values
#trainR.pop('y')
#trainR.pop('ID')
#feature1 = trainR
#target2 = testR["y"].values
#testR.pop('y')
#testR.pop('ID')
#feature2 = testR
#
#""" this is to test best tree depth"""
#for i in range(1,50):
#    regr_1=GradientBoostingRegressor(n_estimators=30, learning_rate=1.0,max_depth=i, random_state=0)
#    regr_1.fit(feature1,target1)
#    y_1=regr_1.predict(feature2)
#    s1=r2_score(target2,y_1)
#    print(i)
#    print(s1)
"""END"""


target1 = train["y"].values

train.pop('y')
train.pop('ID')


feature1 = train
#regr_1=RandomForestRegressor(max_depth=5)
regr_1=GradientBoostingRegressor(n_estimators=30, learning_rate=1.0,max_depth=1, random_state=0)
regr_1.fit(feature1,target1)


column=['X0','X1','X2','X3','X4','X5','X6','X8']
for i in column:
    input = testKag[[i]]
    output = []
    for k in input[i]:
        num=sum([ord(j) for j in k],0)
        number = num-96
        output.append(number)
    output=[int(m) for m in output]
    df = pd.DataFrame(np.array(output).reshape(4209,1), columns = list("a"))
    testKag[i]=df['a']

id=testKag['ID']
testKag.pop('ID')
feature2 = testKag
y_1=regr_1.predict(feature2)
data_frame = pd.DataFrame(id,columns=['ID'])
data_frame['y'] = pd.Series(y_1, index=data_frame.index)
data_frame['ID'] = data_frame["ID"].astype(int)
np.savetxt('submission8.csv',data_frame,delimiter=',')



