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

start_time  = time.time()

#Stores the RMSE values
SqError = []
#Read from the file
df = pd.read_csv("SS-B.csv")



#FOR ONE HOT ENCODER
x = df.drop(['<$a','<$b'], axis ='columns')
print(x)
##################################
y = df['<$b'].values
print(y.shape)



#FOR INTEGER ENCODER
##################################################################
##label_encoder = preprocessing.LabelEncoder()
##df['$a'] = label_encoder.fit_transform(df['$a'])
##df['$b'] = label_encoder.fit_transform(df['$b'])
##df['$c'] = label_encoder.fit_transform(df['$c'])
##data = df.drop(['<$b','<$a'], axis ='columns').values
######################################################################


#The Hyperparameter optimisation for the ML algorithm 
##values = np.arange(10,100,20)
##RF = RandomForestRegressor()
##param = {'n_estimators':values,
##         'criterion':('mse','mae'),
##         'max_features':('auto','sqrt','log2'),
##         'min_samples_split':[2,3,4,5,6]
##         }

#ONE HOT ENCODER
#######################################################
column_trans = make_column_transformer(
        (OneHotEncoder(), ['$a','$b','$c']))
print(column_trans.fit_transform(x.head()))

data = column_trans.fit_transform(x)
############################################################


##knn = KNeighborsRegressor()
##param={'n_neighbors': [3,5,7,10,11,15],
##       'weights': ('uniform','distance'),
##       'algorithm': ('auto','ball_tree','kd_tree'),
##       'leaf_size': [10,20,30,40,50,60,70,80,90],
##       }

values = np.arange(0.01,5,0.5)
svr = SVR()
param = {'kernel': ('linear','poly','rbf','sigmoid'),
         'degree': [2,3,4,5],
         'gamma': ('scale','auto'),
         'coef0': [0,1,2,3,4,5,6,7,8,9,10],
         'epsilon':values
         }


##DT = DecisionTreeRegressor()
##param = {'criterion' : ('mse','friedman_mse','mae','poisson'),
##         'splitter': ('best','random'),
##         'min_samples_split': [0.1,0.2,0.5,1,1.5,2,2.5,3,4,5,6,7]
##         }

##LR = LinearRegression()
##param = {'fit_intercept':('True','False'),
##         'normalize':('True','False'),
##         'n_jobs':[1,-1]
##         }



##x=np.arange(0.1,10,0.5)
##print(x)
##KR = KernelRidge()
##param = {'alpha':x,
##         'degree':[1,2,3,4,5,6,7,8,9],
##         'coef0':[1,2,3,4,5]
##         }

#The training and testing split function 
rkf = RepeatedKFold(n_splits=10, n_repeats=5, random_state=1)
for train_index, test_index in rkf.split(data):
    print("train:", train_index, "test:", test_index)
    #Puts the values into the function
    gridS = GridSearchCV(svr, param)
    gridS.fit(data[train_index],y[train_index])
    print("Execution time: " + str((time.time() - start_time)) + ' seconds')
    print(gridS.best_params_)
    #With the training data fitted, its tests it against the remaining values
    #and  prints out the 50 results
    y_pred = cross_val_predict(SVR(**gridS.best_params_),data[test_index],y[test_index],cv=5)
    sqe = math.sqrt(mean_squared_error(y[test_index],y_pred))
    SqError.append(sqe)


print(SqError)



