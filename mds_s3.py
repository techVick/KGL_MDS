"""TechVick_GonnaWin"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
#from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
import matplotlib.pylab as plt

print("hello")
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
trainR, testR = train_test_split(train, test_size = 0.2)

target = trainR["y"].values
trainR.pop('y')
trainR.pop('ID')
predictors = trainR
target2 = testR["y"].values
testR.pop('y')
testR.pop('ID')
feature2 = testR

import scipy.stats as st

one_to_left = st.beta(10, 1)  
from_zero_positive = st.expon(0, 50)

params = {  
    "n_estimators": st.randint(3, 40),
    "max_depth": st.randint(3, 40),
    "learning_rate": st.uniform(0.05, 0.4),
    "colsample_bytree": one_to_left,
    "subsample": one_to_left,
    "gamma": st.uniform(0, 10),
    'reg_alpha': from_zero_positive,
    "min_child_weight": from_zero_positive,
}

xgbreg = XGBRegressor()
from sklearn.model_selection import RandomizedSearchCV
gs = RandomizedSearchCV(xgbreg, params, n_jobs=1)  
gs.fit(predictors, target) 
g=gs.predict(feature2)


#xclas = XGBRegressor() 
#xclas.fit(predictors, target)  
#g=xclas.predict(feature2)


#T_train_xgb = xgb.DMatrix(predictors, target)
#params = {"objective": "reg:linear", "booster":"gblinear"}
#gbm = xgb.train(dtrain=T_train_xgb,params=params)
#Y_pred = gbm.predict(xgb.DMatrix(pd.DataFrame(feature2)))



s1=r2_score(target2,g)
print(s1)

