#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd
import math
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


# #############################################################################
# Import the sample data

# mappings: the combinations of strategies
dd_x = 'mappings_10_stages.csv'
sample_x = pd.read_csv(dd_x, header = None)
sample_x = sample_x.to_numpy()

# the two objectives
dd_y = 'new_re_cal_10_stages.csv'
sample_y = pd.read_csv(dd_y, header = None)
sample_y = sample_y.to_numpy()

X = sample_x

# first objective: cost
y1 = sample_y[:,0]
y1 = np.reshape(y1,(len(sample_x),1))
# second objective: damage
y2 = sample_y[:,1]

y2 = np.reshape(y2,(len(sample_x),1))




# #############################################################################
# Import data to be predicted
strategy_1 = pd.read_csv('strategies_c1.csv', header = None)
strategy_1 = strategy_1.to_numpy()
strategy_2 = pd.read_csv('strategies_c2.csv', header = None)
strategy_2 = strategy_2.to_numpy()
strategy_3 = pd.read_csv('strategies_c3.csv', header = None)
strategy_3 = strategy_3.to_numpy()



# all of the mappings
n_rbf = 4
n_vars = n_rbf * 3
n_bins = 2
n_beliefs = n_bins + 1
mappings = np.zeros((len(strategy_1) * len(strategy_2) * len(strategy_3), n_vars * n_beliefs))
c = 0
for i in range(len(strategy_1)):
    for j in range(len(strategy_2)):
        for k in range(len(strategy_3)):
            mappings[c,:] = np.concatenate(((strategy_1[i],strategy_2[j], strategy_3[k])))
            c = c + 1


from sklearn.neural_network import MLPRegressor


def tune_hyperparameter_nn(x, y, random_state, max_iter, tol):
    fold = 3
    # train the algorithm
    # hyperparameter, needs to be tuned later

    accuracy_test = np.zeros(fold)
    m=0


    for k in range(fold):
#                 test_set = np.array_split(data, fold)[k]
#                 train_set = data.drop(test_set.index)
#                 x_train = train_set.drop(['match'], axis=1)
#                 y_train = train_set['match']
#                 x_test = test_set.drop(['match'], axis=1)
#                 y_test = test_set['match']
        x = pd.DataFrame(x)
        y = pd.DataFrame(y)
        x_test = np.array_split(x, fold)[k]
        y_test = np.array_split(y, fold)[k]
        x_train = x.drop(x_test.index)
        y_train = y.drop(y_test.index)

        clf = MLPRegressor(random_state = 42, max_iter = 1000, tol = 0.00001)
        #clf = make_pipeline(StandardScaler(), SVR(C = C[i], kernel = 'rbf',gamma=gamma[j]))
        clf.fit(x_train, y_train)
        pred_test = clf.predict(x_test)
        pred_test = np.reshape(pred_test,(-1,1))

        accuracy_test[k] = math.sqrt((((y_test - pred_test) ** 2).sum())/len(sample_y))
    m=m+1
    accuracy_test_mean = accuracy_test.mean()
    accuracy_test_ste= accuracy_test.std()/len(accuracy_test)


    return accuracy_test_mean, accuracy_test_ste


###################################################
##### The following part needs to be repeated #####
##### several times to control the randomness #####
###################################################
# control the random seed
seed = 159
pred_x = mappings

# #############################################################################
# Fit regression model

clf_y1 = MLPRegressor(random_state=seed, max_iter=1000, tol = 0.00001)
clf_y1.fit(X, y1)

pred_y1 = clf_y1.predict(pred_x)

clf_y2 = MLPRegressor(random_state=seed, max_iter=1000, tol = 0.00001)
clf_y2.fit(X, y2)

pred_sample_y2 = clf_y2.predict(X)
pred_sample_y2 = np.reshape(pred_sample_y2,(-1,1))
accuracy_y2 = math.sqrt((((y2 - pred_sample_y2) ** 2).sum())/len(sample_y))


pred_y2 = clf_y2.predict(pred_x)


# filter the predictions smaller than 0

pred_y1 = np.reshape(pred_y1, (-1,1))
pred_y2 = np.reshape(pred_y2, (-1,1))
mid_y = np.concatenate((pred_y1,pred_y2), axis = 1)
y = mid_y[mid_y.min(axis=1)>=0,:]


np.savetxt('pred_10_stages_' + str(seed) + '.csv',y, delimiter =",", fmt = '% s')


###### Run pareto.py to sort solution before running following lines #####

# read the solutions on Pareto frontier
dd_pp = 'parsed_pred_10_stages_' + str(seed) + ".csv"
pp = pd.read_csv(dd_pp, header = None)
pp = pp.to_numpy()
l = len(pp)
mp = np.zeros((l, 36))


# find the corresponding meta-policies
for i in range(l):
    y1_loc = np.isclose(pred_y1, pp[i,0], rtol=1.e-15, atol=1.e-15).nonzero()[0][0]
    y2_loc = np.isclose(pred_y2, pp[i,1], rtol=1.e-15, atol=1.e-15).nonzero()[0][0]
    if y1_loc != y2_loc:
        print ("Error: The two indexes are not equal")
    mp[i] = mappings[y1_loc]
    
# save the mp with seed
np.savetxt('mp_pred_10_stages_' + str(seed)+ '.csv', mp, delimiter =",", fmt = '% s')






