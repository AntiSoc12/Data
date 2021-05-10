import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import math
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.utils import resample
from sklearn.model_selection import RepeatedKFold
from scipy.stats import wilcoxon
import time
from sklearn.neural_network import MLPRegressor


SqError = []
df = pd.read_csv("SS-C.csv")



print(df)

#FOR ONE HOT ENCODER
##x = df.drop(['<$throughput','<$latency','spliters','counters'], axis ='columns')
##print(x)
##################################
y = df['<$throughput'].values
#y = y.astype(int)
print(y.shape)
##xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=1/3, random_state=42)


##print(xtrain.head())
##print(ytrain.shape)

##df.columns=df.columns.str.replace('$','')
#FOR LABEL ENCODER
##################################################################
label_encoder = preprocessing.LabelEncoder()
df['spout_wait'] = label_encoder.fit_transform(df['spout_wait'])
data = df.drop(['<$throughput','<$latency','spliters','counters'], axis ='columns').values
######################################################################
##print(data)
##print(data[1])
##
##values = np.arange(10,100,20)
##RF = RandomForestRegressor()
##param = {'n_estimators':values,
##         'criterion':('mse','mae'),
##         'max_features':('auto','sqrt','log2'),
##         'min_samples_split':[2,3,4,5,6]
##         }
##gridS = GridSearchCV(RF, param, cv=5)
##gridS.fit(column_trans.fit_transform(train_index),train_index)
##print(gridS.best_params_)


##def OneHot():
#######################################################
##column_trans = make_column_transformer(
##        (OneHotEncoder(), ['spout_wait']))
##print(column_trans.fit_transform(x.head()))
##
##data = column_trans.fit_transform(x)
############################################################

##
##knn = KNeighborsRegressor()
##param={'n_neighbors': [3,5,7,10,11,15],
##       'weights': ('uniform','distance'),
##       'algorithm': ['auto'],#'ball_tree','kd_tree'),
##       'leaf_size': [10,20,30,40,50,60,70,80,90],
##       }
##gridS = GridSearchCV(knn, param, cv=5)
##gridS.fit(column_trans.fit_transform(xtrain),ytrain)
##print(gridS.best_params_)

values = np.arange(0.01,5,0.5)
svr = SVR()
param = {'kernel': ('linear','rbf'),
         'degree': [2,3,4,5],
         'gamma': ('scale','auto'),
         'coef0': [0,1,2,3,4,5,6,7,8,9,10],
         'epsilon':[0.01,1]
         }
##gridS = GridSearchCV(svr, param, cv=5)
##gridS.fit(column_trans.fit_transform(xtrain),ytrain)
##print(gridS.best_params_)


##DT = DecisionTreeRegressor()
##param = {'criterion' : ('mse','friedman_mse','mae'),
##         'splitter': ('best','random'),
##         'min_samples_split': [0.1,0.2,0.5,1.0,2,3,4,5,6,7]
##         }
##gridS = GridSearchCV(DT, param, cv=5)
##gridS.fit(column_trans.fit_transform(xtrain),ytrain)
##print(gridS.best_params_)
##
##LR = LinearRegression()
##param = {'fit_intercept':('True','False'),
##         'normalize':('True','False'),
##         'n_jobs':[1,-1]
##         }
##gridS = GridSearchCV(LR, param, cv=5)
##gridS.fit(column_trans.fit_transform(xtrain),ytrain)
##print(gridS.best_params_)

##mlp = MLPRegressor()
##param = {'alpha':[0.000001,0.0001,0.01,0.1,1],
##         'activation': ('identity','logistic','tanh','relu'),
##         'max_iter':[10,20,30,40,100,150,200]
##         }



##x1=np.arange(0.1,5,0.5)
##KR = KernelRidge()
##param = {'alpha':x1,
##         'degree':[1,2,3,4,5,6,7,8,9],
##         'coef0':[1,2,3,4,5]
##         }
##gridS = GridSearchCV(KR, param, cv=5)
##gridS.fit(column_trans.fit_transform(xtrain),ytrain)
##print(gridS.best_params_)

rkf = RepeatedKFold(n_splits=10, n_repeats=5, random_state=1)
for train_index, test_index in rkf.split(data):
    #print("train:", train_index, "test:", test_index)
    start_time  = time.time()
    gridS = GridSearchCV(svr, param)
    gridS.fit(data[train_index],y[train_index])
    #print(gridS.best_params_)
    y_pred = cross_val_predict(SVR(**gridS.best_params_),data[test_index],y[test_index],cv=5)
    sqe = math.sqrt(mean_squared_error(y[test_index],y_pred))
    SqError.append(sqe)
    print("Execution time: " + str((time.time() - start_time)) + ' seconds')



print(SqError)
w,p = wilcoxon(SqError)
print(w,p)



