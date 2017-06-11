import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import AdaBoostRegressor
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
        ggg='{0:07b}'.format(number)
        output.append(ggg)
#        print(output)
    output=[int(m) for m in output]
    df = pd.DataFrame(np.array(output).reshape(4209,1), columns = list("a"))
    train[i]=df['a']


for i in column:
    train[i]=train[i].astype(str)
    for j in range(7):
        k=j+300
        train[i+str(k)] = train[i].str[j]
        train.fillna(0, inplace=True)
        train[i+str(k)]=train[i+str(k)].astype(int)
train.drop(column,inplace=True,axis=1)

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
#s2=0
#""" this is to test best tree depth"""
##j=np.arange(5,425,5)
##for i in np.nditer(j):
#for i in range(1,300,10):
#    regr_1=GradientBoostingRegressor(n_estimators=i, learning_rate=0.1,max_depth=1, random_state=0)
##    regr_1=AdaBoostRegressor(n_estimators=i)
#    regr_1.fit(feature1,target1)
#    y_1=regr_1.predict(feature2)
#    s1=r2_score(target2,y_1)
#    print(i)
#    print(s1)
#    if s1>s2:
#       treeDepth=i
#       s2=s1
"""END"""

target1 = train["y"].values

train.pop('y')
train.pop('ID')


feature1 = train
regr_1=GradientBoostingRegressor(n_estimators=271, learning_rate=0.1,max_depth=1, random_state=0)
regr_1.fit(feature1,target1)


column=['X0','X1','X2','X3','X4','X5','X6','X8']
for i in column:
    input = testKag[[i]]
    output = []
    for k in input[i]:
        num=sum([ord(j) for j in k],0)
        number = num-96
        ggg='{0:07b}'.format(number)
        output.append(ggg)
    output=[int(m) for m in output]
    df = pd.DataFrame(np.array(output).reshape(4209,1), columns = list("a"))
    testKag[i]=df['a']
for i in column:
    testKag[i]=testKag[i].astype(str)
    for j in range(7):
        k=j+300
        testKag[i+str(k)] =testKag[i].str[j]
        testKag.fillna(0, inplace=True)
        testKag[i+str(k)]=testKag[i+str(k)].astype(int)
testKag.drop(column,inplace=True,axis=1)


id=testKag['ID']
testKag.pop('ID')
feature2 = testKag
y_1=regr_1.predict(feature2)
data_frame = pd.DataFrame(id,columns=['ID'])
data_frame['y'] = pd.Series(y_1, index=data_frame.index)
data_frame['ID'] = data_frame["ID"].astype(int)
np.savetxt('submission9.csv',data_frame,delimiter=',')
