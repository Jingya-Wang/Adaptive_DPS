#!/usr/bin/env python
# coding: utf-8



'''
DPS project;
Evaluate the performance of meta-policies

'''

import numpy as np
import pandas as pd
import math
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from statistics import NormalDist
from scipy.stats import norm

import timeit
start = timeit.default_timer()

dd_1 = '../sampled_water_level_10000.csv'
dd = '../sampled_water_level_100.csv'
water_level = pd.read_csv(dd, header = None)
water_level = water_level.to_numpy()

all_water_level = pd.read_csv(dd_1, header = None)
all_water_level = all_water_level.to_numpy()


# divide SOWs into two bins
final_water = all_water_level[:,329]
final_water = np.sort(final_water)
bin_1 = np.array([-100000, final_water[4999]])
bin_2 = np.array([final_water[5000],100000])
bin_matrix = [bin_1, bin_2]
print(bin_1, bin_2)

# assign each bins probabilities

w_1 = 0.5
w_2 = 0.3
w_3 = 0.7

# read in generated strategies (interval sampled for computational efficiency)

strategy_1 = pd.read_csv('sampled_str_c1.csv', header = None)
strategy_1 = strategy_1.to_numpy()

strategy_2 = pd.read_csv('sampled_str_c2.csv', header = None)
strategy_2 = strategy_2.to_numpy()

strategy_3 = pd.read_csv('sampled_str_c3.csv', header = None)
strategy_3 = strategy_3.to_numpy()

# construct bins
n_bins = 2
n_beliefs = n_bins + 1
beliefs_space = np.array([[w_1, 1-w_1],[w_2, 1-w_2],[w_3, 1-w_3]])


# input parameters I
n = 30
year_offset = 30
t_years = 330
n_years = 300
years = np.arange(0,n_years,1)
#years_1 = np.arange(0,301,1)
base_year = 2015
sl2000 = 0.0
time_step = 100
time_stage = int(n_years/time_step)
#print(len(water_level))

n_obj = 2
n_rbf = 4
n_vars = n_rbf * 3

## parameters II
V_0 = 22656.5 # loss due to flood prior to t=0
alpha = 0.0574
p_0 = 0.00110  #
h_0 = (1/alpha)*(np.log(1/p_0)) # initial height of dike prior to t=0
gamma = 0.02  # parameter for calculating damage
zeta = 0.002032  # parameter for calculating damage
kappa = 324.6287  # parameter for calculating damage
upsilon = 2.1304  # parameter for calculating damage
lam = 0.01  # parameter for calculating damage
delta = 0.04   # discount rate for investment
delta_1 = 0.04  # discount rate for damages

# SOW
n_sow = 100
inv_n_sow = 1/n_sow

# constraint
rel_threshold = 0.8


# Hyperparameter for locally weighted regression

M = 1
tao = 10


############ locally weighted regression#################
############ use the past year data to predict; integrate the probabilities#################

def Closed_form_linear_weighted(query, data_x,data_y,M,tao):
    
    n_data = len(data_x)
    phi_matrix = np.zeros((n_data,M+1))
    R_matrix = np.zeros((n_data,n_data))
    wts = np.zeros(n_data)
    for i in range (n_data):
        for j in range(M+1):
            phi_matrix[i,j] = data_x[i]**j
    for i in range (n_data):
        R_matrix[i,i] = np.exp(-((query-data_x[i])**2)/2/tao**2)/2
        wts[i] = R_matrix[i,i]
    phi_matrix_transpose = np.transpose(phi_matrix)
    phi_R_product = np.matmul(np.matmul(phi_matrix_transpose,R_matrix),phi_matrix)
    phi_R_product_inv = np.linalg.inv(phi_R_product)
    w = np.matmul(np.matmul(np.matmul(phi_R_product_inv,phi_matrix_transpose),R_matrix),data_y)
    
    return w, wts

def get_probs(time_stage, years, time_step, new_water_level, n_years, bin_matrix):
    probs = np.zeros((n_bins, time_stage))
    
    for i in range(time_stage):
        
        # locally weighted regression
        lwr_years = years[time_step * i : time_step * (i+1)]
        lwr_years = lwr_years.reshape(-1,1)
        lwr_water_level = new_water_level[time_step * i : time_step * (i+1)]
        lwr_water_level = lwr_water_level.reshape(-1,1)

        x_predict_weighted = lwr_years
        y_predict_weighted = np.zeros(len(x_predict_weighted))
        x_feature = np.zeros(M+1)
        for j in range(len(x_predict_weighted)):
            w_closed_form_weighted_1, wts_1 = Closed_form_linear_weighted(x_predict_weighted[j], lwr_years, lwr_water_level, M, tao)
            for m in range(M+1):
                x_feature[m] = x_predict_weighted[j]**m
            y_predict_weighted[j] = np.matmul(x_feature,w_closed_form_weighted_1)
        y_predict_weighted = y_predict_weighted.reshape(-1,1)
        w_closed_form_weighted, wts = Closed_form_linear_weighted(n_years, lwr_years, lwr_water_level, M, tao)
        for m in range(M+1):
            x_feature[m] = n_years**m
        final_predict_weighted = np.matmul(x_feature,w_closed_form_weighted)

        wtd_mse = np.average((lwr_water_level - y_predict_weighted)**2, axis =0, weights = wts)

        nd = NormalDist(mu=final_predict_weighted, sigma=wtd_mse)
        for l in range(n_bins):

            probs[l,i] = nd.cdf(bin_matrix[l][1]) - nd.cdf(bin_matrix[l][0])

    print(probs)

    return probs
    
    
# pick up strategies based on probs
def pickup_strategy(strategy_1, strategy_2, strategy_3, probs, beliefs_space, time_stage_ind):
    distances = np.zeros(n_beliefs)
    if time_stage_ind == 0:
        strategy = strategy_1
        strategy_name = 's1'
    else:
        for i in range(n_beliefs):
            distances[i] = sum([abs(j-k) for j,k in zip(probs,beliefs_space[i,:])])

        ind = int(np.where(distances == min(distances))[0]) + 1

        strategy_name = 's' + str(ind)
        full_strategy_name = 'strategy_' + str(ind)
        
        strategy = globals()[full_strategy_name]

    return strategy, strategy_name
    

####################################
## get dike heightening for time t
## based on calculating state varaibles, \
## BH and FH, and then heightening
####################################

def calc_heightening(t, prev_h, water_level, n, year_offset, x, r, w, h_0):

    # set the previous height as h_0; otherwise, set it as the previous timestep
    prev_sl = water_level[t-1+year_offset]
    
    ####################################
    ## fit the simple linear regression based on
    ## past n-year water level data. The coefficients
    ## are used as state variables to describe states##
    ####################################
    
    def state_variables(past_n_year_data, n):
        x = np.arange(0,n)
        x = x.reshape(-1,1)
        y = past_n_year_data
        y = y.reshape(-1,1)
        linear_reg = LinearRegression()
        linear_reg.fit(x, y)
        coef = linear_reg.coef_
        pred = linear_reg.predict(x)
        ssr = mean_squared_error(y,pred) * len(y)
        srss = math.sqrt(ssr)

        return coef, srss
    
    obs = np.zeros(n)
    for i in range(n):
        obs[i] = water_level[t+year_offset-n+i]
    mean_slr_rate, srss = state_variables(obs,n)

    ####################################
    ## calculate the freeboard height and
    ## buffer height for time t with given DPS decision variables##
    ####################################
    FH_t = 0
    BH_t = 0
    FH_t += x[0] + r[0] * mean_slr_rate + w[0] * mean_slr_rate * mean_slr_rate
    FH_t += x[1] + r[1] * srss + w[1] * srss * srss
    BH_t += x[2] + r[2] * mean_slr_rate + w[2] * mean_slr_rate * mean_slr_rate
    BH_t += x[3] + r[3] * srss + w[3] * srss * srss

    if FH_t < 0:
        FH_t = 0
        
    if BH_t < 0:
        BH_t = 0
        
    ####################################
    ## calculate dike heightenings##
    ####################################
    test_height = prev_h - BH_t
    u_t = 0
    if prev_sl > test_height:
        safe_height = prev_sl - test_height
        u_t = safe_height + FH_t

    return u_t


####################################
## calculate expected loss and investment
## at time t
####################################
def calc_inv_loss(t, prev_h, u_t, water_level, year_offset, V_0, gamma, zeta, h_0, kappa, upsilon, lam):
    
    flood_rel = 0
    current_h = prev_h + u_t

    # expected loss
    V_t = V_0 * math.exp(gamma * t) * math.exp(zeta * (current_h - h_0))
    
    if current_h <= water_level[t + year_offset]:
        s_t = V_t
        flood_rel = 1
    else:
        s_t = 0
            
    # investment
    if u_t == 0:
        inv_t = 0
    else:
        inv_t = (kappa + upsilon * u_t) * math.exp(lam * (current_h - h_0))
        
    return inv_t, s_t, current_h, flood_rel


####################################
## calculate discounted loss and investment
## over time
####################################
def discounted(inv, s, n_years, delta, delta_1):
    
    # discounted investment
    total_inv = 0
    total_s = 0
    for i in range(n_years):
        total_inv += inv[i] * math.exp(-1 * delta * i)
        total_s += s[i] * math.exp(-1 * delta_1 * i)
        
    return total_inv, total_s



####################################
## calculate the model for all time steps
####################################
def calc_model(n_years, h_0, water_level, n, year_offset, x, r, w, V_0,              gamma, zeta, kappa, upsilon, lam, delta, delta_1):
    inv = np.zeros(n_years)
    s = np.zeros(n_years)
    FH = np.zeros(n_years)
    BH = np.zeros(n_years)
    u = np.zeros(n_years)
    flood_rel = np.zeros(n_years)
    
    for i in range(n_years):
        
        if i == 0:
            prev_h = h_0

        u[i] = calc_heightening(i, prev_h, water_level, n, year_offset, x, r, w, h_0)

        inv[i], s[i], current_h, flood_rel[i] = calc_inv_loss(i, prev_h, u[i], water_level, year_offset, V_0, gamma, zeta, h_0, kappa, upsilon, lam)

        prev_h = current_h

    total_inv, total_s = discounted(inv, s, n_years, delta, delta_1)
    
    return total_inv, total_s, flood_rel



def selections(time_stage, years, time_step, new_water_level, n_years, bin_matrix):
    probs = get_probs(time_stage, years, time_step, new_water_level, n_years, bin_matrix)
    print(probs)
    #the decision for first time stage is just the combination 1
    timestage1, timestage1_name = pickup_strategy(strategy_1, strategy_2, strategy_3, probs[:,0],beliefs_space, 0)
    #the decision for second time stage is based on the first prediction,
    timestage2, timestage2_name = pickup_strategy(strategy_1, strategy_2, strategy_3, probs[:,0], beliefs_space, 1)
    #the decicion for third time stage is based on the second prediction (the third prediction actually doesn't matter)
    timestage3, timestage3_name = pickup_strategy(strategy_1, strategy_2, strategy_3, probs[:,1], beliefs_space, 2)
    timestage4, timestage4_name = pickup_strategy(strategy_1, strategy_2, strategy_3, probs[:,2], beliefs_space, 3)
    timestage5, timestage5_name = pickup_strategy(strategy_1, strategy_2, strategy_3, probs[:,3], beliefs_space, 4)
    timestage6, timestage6_name = pickup_strategy(strategy_1, strategy_2, strategy_3, probs[:,4], beliefs_space, 5)
    return timestage1, timestage2, timestage3, timestage4, timestage5, timestage6,\
timestage1_name, timestage2_name, timestage3_name, timestage4_name, timestage5_name, timestage6_name

# the same as in 3_stage case; so, this has been done
#mappings_name = np.zeros((len(strategy_1) * len(strategy_2) * len(strategy_3), n_vars * n_beliefs))
#c = 0
#for i in range(len(strategy_1)):
#    for j in range(len(strategy_2)):
#        for k in range(len(strategy_3)):
#            mappings_name[c,:] = np.concatenate(((strategy_1[i],strategy_2[j], strategy_3[k])))
#            c = c + 1
#
#np.savetxt('mappings_3_stages.csv',mappings_name, delimiter =",", fmt = '% s')


def total_inv_s(timestage1, timestage2, timestage3, timestage4, timestage5, timestage6, one_water_level):
    inv = np.zeros(n_years)
    s = np.zeros(n_years)
    for t in range(0,50):
        rbf_vars = timestage1
        x = rbf_vars[0::3]
        r = rbf_vars[1::3]
        w = rbf_vars[2::3]
        if t == 0:
            prev_h = h_0
        u = calc_heightening(t, prev_h, one_water_level, n, year_offset, x, r, w, h_0)
        inv[t], s[t], current_h, flood_rel = calc_inv_loss(t, prev_h, u, one_water_level, year_offset, V_0, gamma, zeta, h_0, kappa, upsilon, lam)
        prev_h = current_h
    for t in range(50,100):
        rbf_vars = timestage2
        x = rbf_vars[0::3]
        r = rbf_vars[1::3]
        w = rbf_vars[2::3]
        u = calc_heightening(t, prev_h, one_water_level, n, year_offset, x, r, w, h_0)
        inv[t], s[t], current_h, flood_rel = calc_inv_loss(t, prev_h, u, one_water_level, year_offset, V_0, gamma, zeta, h_0, kappa, upsilon, lam)
        #current_h = prev_h + u[count,t]
        prev_h = current_h
    for t in range(100,150):
        rbf_vars = timestage3
        x = rbf_vars[0::3]
        r = rbf_vars[1::3]
        w = rbf_vars[2::3]
        u = calc_heightening(t, prev_h, one_water_level, n, year_offset, x, r, w, h_0)
        inv[t], s[t], current_h, flood_rel = calc_inv_loss(t, prev_h, u, one_water_level, year_offset, V_0, gamma, zeta, h_0, kappa, upsilon, lam)
        #current_h = prev_h + u[count,t]
        prev_h = current_h
    for t in range(150,200):
        rbf_vars = timestage4
        x = rbf_vars[0::3]
        r = rbf_vars[1::3]
        w = rbf_vars[2::3]
        u = calc_heightening(t, prev_h, one_water_level, n, year_offset, x, r, w, h_0)
        inv[t], s[t], current_h, flood_rel = calc_inv_loss(t, prev_h, u, one_water_level, year_offset, V_0, gamma, zeta, h_0, kappa, upsilon, lam)
        #current_h = prev_h + u[count,t]
        prev_h = current_h
    for t in range(200,250):
        rbf_vars = timestage5
        x = rbf_vars[0::3]
        r = rbf_vars[1::3]
        w = rbf_vars[2::3]
        u = calc_heightening(t, prev_h, one_water_level, n, year_offset, x, r, w, h_0)
        inv[t], s[t], current_h, flood_rel = calc_inv_loss(t, prev_h, u, one_water_level, year_offset, V_0, gamma, zeta, h_0, kappa, upsilon, lam)
        #current_h = prev_h + u[count,t]
        prev_h = current_h
    for t in range(250,300):
        rbf_vars = timestage6
        x = rbf_vars[0::3]
        r = rbf_vars[1::3]
        w = rbf_vars[2::3]
        u = calc_heightening(t, prev_h, one_water_level, n, year_offset, x, r, w, h_0)
        inv[t], s[t], current_h, flood_rel = calc_inv_loss(t, prev_h, u, one_water_level, year_offset, V_0, gamma, zeta, h_0, kappa, upsilon, lam)
        #current_h = prev_h + u[count,t]
        prev_h = current_h
    total_inv, total_v = discounted(inv, s, n_years, delta, delta_1)

    return total_inv, total_v


def find_set_inds(mappings, mapping_name, strategy_1, strategy_2, strategy_3):
 results = np.zeros(len(strategy_1) * len(strategy_2) * len(strategy_3))
 new_str_2 = set(mapping_name.split(","))
 for i in range(len(mappings)):
     new_str = set(mappings_array[i].split(","))
     results[i] = new_str_2.issubset(new_str)
 ind = np.where(results == True)
 ind = np.array(ind)
 ind = ind.T
 return ind


mapping_space = np.zeros((len(strategy_1) * len(strategy_2) * len(strategy_3), len(water_level) * n_obj))
mappings = []
for i in range(len(strategy_1)):
 S1 = 'S1'
 for j in range(len(strategy_2)):
     S2 = 'S2'
     for k in range(len(strategy_3)):
         S3 = 'S3'
         mappings.append(S1 + '_' + str(i+1) + S2 + '_' + str(j+1) + S3 + '_' + str(k+1))
mappings_array = np.array(mappings)
mappings_dict = {'mappings': mappings}
df = pd.DataFrame(mappings)



def mapping(sow, one_water_level,timestage1_name,timestage2_name,timestage3_name, timestage4_name,timestage5_name,timestage6_name,\
            timestage1, timestage2, timestage3, timestage4, timestage5,timestage6):
    print(timestage1_name, timestage2_name, timestage3_name, timestage4_name, timestage5_name, timestage6_name)
    ss = {timestage1_name, timestage2_name, timestage3_name, timestage4_name, timestage5_name, timestage6_name}
    ss = np.array(np.sort(list(ss)))
    print(ss)
    if len(ss) == 3:
    # ABC
        for i in range(len(strategy_1)):
            for j in range(len(strategy_2)):
                for k in range(len(strategy_3)):
                    ind_1 = i
                    if timestage2_name == 'strategy_1':
                        ind_2 = i
                    elif timestage2_name == 'strategy_2':
                        ind_2 = j
                    else:
                        ind_2 = k
                    if timestage3_name == 'strategy_1':
                        ind_3 = i
                    elif timestage3_name == 'strategy_2':
                        ind_3 = j
                    else:
                        ind_3 = k
                    if timestage4_name == 'strategy_1':
                        ind_4 = i
                    elif timestage4_name == 'strategy_2':
                        ind_4 = j
                    else:
                        ind_4 = k
                    if timestage5_name == 'strategy_1':
                        ind_5 = i
                    elif timestage5_name == 'strategy_2':
                        ind_5 = j
                    else:
                        ind_5 = k
                    if timestage6_name == 'strategy_1':
                        ind_6 = i
                    elif timestage6_name == 'strategy_2':
                        ind_6 = j
                    else:
                        ind_6 = k
                    
                    total_inv, total_s = total_inv_s(timestage1[ind_1,:], timestage2[ind_2,:], timestage3[ind_3,:],\
                                                    timestage4[ind_4,:], timestage5[ind_5,:], timestage6[ind_6,:], one_water_level)

                    mapping_name = 'S_1' + '_' + str(i+1) + ',' + 'S_2' + '_' + str(j+1) + ',' + 'S_3' + '_' + str(k+1)

                    ind = np.where(mappings_array == mapping_name)

                    mapping_space[ind, sow * n_obj] = total_inv
                    mapping_space[ind, sow * n_obj + 1] = total_s
    
    if len(ss) == 2 and ss[1] == 'strategy_3':
        # AC
        for i in range(len(strategy_1)):
            for j in range(len(strategy_3)):
                ind_1 = i
                if timestage2_name == 'strategy_1':
                    ind_2 = i
                else:
                    ind_2 = j
                if timestage3_name == 'strategy_1':
                    ind_3 = i
                else:
                    ind_3 = j
                if timestage4_name == 'strategy_1':
                    ind_4 = i
                else:
                    ind_4 = j
                if timestage5_name == 'strategy_1':
                    ind_5 = i
                else:
                    ind_5 = j
                if timestage6_name == 'strategy_1':
                    ind_6 = i
                else:
                    ind_6 = j

                total_inv, total_s = total_inv_s(timestage1[ind_1,:], timestage2[ind_2,:], timestage3[ind_3,:],\
                                                        timestage4[ind_4,:], timestage5[ind_5,:], timestage6[ind_6,:], one_water_level)

                mapping_name = 'S_1' + '_' + str(i+1) + ',' + 'S_3' + '_' + str(j+1)

                ind = find_set_inds(mappings_array, mapping_name, strategy_1, strategy_2, strategy_3)

                for m in range(len(ind)):
                    mapping_space[ind[m], sow * n_obj] = total_inv
                    mapping_space[ind[m], sow * n_obj + 1] = total_s
    
    if len(ss) == 2 and ss[1] == 'strategy_2':
        # AB
        for i in range(len(strategy_1)):
            for j in range(len(strategy_2)):
                ind_1 = i
                if timestage2_name == 'strategy_1':
                    ind_2 = i
                else:
                    ind_2 = j
                if timestage3_name == 'strategy_1':
                    ind_3 = i
                else:
                    ind_3 = j
                if timestage4_name == 'strategy_1':
                    ind_4 = i
                else:
                    ind_4 = j
                if timestage5_name == 'strategy_1':
                    ind_5 = i
                else:
                    ind_5 = j
                if timestage6_name == 'strategy_1':
                    ind_6 = i
                else:
                    ind_6 = j
                    
                   
                total_inv, total_s = total_inv_s(timestage1[ind_1,:], timestage2[ind_2,:], timestage3[ind_3,:],\
                                                        timestage4[ind_4,:], timestage5[ind_5,:], timestage6[ind_6,:], one_water_level)

                mapping_name = 'S_1' + '_' + str(i+1) + ',' + 'S_2' + '_' + str(j+1)

                ind = find_set_inds(mappings_array, mapping_name, strategy_1, strategy_2, strategy_3)
  
                for m in range(len(ind)):
                    mapping_space[ind[m], sow * n_obj] = total_inv
                    mapping_space[ind[m], sow * n_obj + 1] = total_s
    if len(ss) == 0:
        # A

        for i in range(len(strategy_1)):
            ind_1 = ind_2 = ind_3 = ind_4 = ind_5 = ind_6 = i
            total_inv, total_s = total_inv_s(timestage1[ind_1,:], timestage2[ind_2,:], timestage3[ind_3,:],\
                                                        timestage4[ind_4,:], timestage5[ind_5,:], timestage6[ind_6,:], one_water_level)
            mapping_name = 'S_1' + '_' + str(i+1)
            ind = find_set_inds(mappings_array, mapping_name, strategy_1, strategy_2, strategy_3)
            for m in range(len(ind)):
                mapping_space[ind[m], sow * n_obj] = total_inv
                mapping_space[ind[m], sow * n_obj + 1] = total_s

    return mapping_space

for i in range(len(water_level)):
    print(i)
    one_water_level = water_level[i,:]
    new_water_level = one_water_level[n:]
    timestage1, timestage2, timestage3, timestage1_name, timestage2_name, timestage3_name = selections(time_stage, years, time_step, new_water_level, n_years, bin_matrix)
    mapping_space = mapping(i, one_water_level,timestage1_name,timestage2_name,timestage3_name,            timestage1, timestage2, timestage3)

inv_ind = np.arange(0,len(water_level) * n_obj, 2)
s_ind = np.arange(1,len(water_level) * n_obj, 2)
final_inv = np.sum(mapping_space[:,inv_ind],axis = 1)/len(water_level)
final_s = np.sum(mapping_space[:,s_ind], axis = 1)/len(water_level)
                           
                           
invAs = np.array((final_inv, final_s))
invAs = invAs.T
np.savetxt('new_re_cal_6_stages.csv',  invAs, delimiter =", ",  fmt ='% s')

stop = timeit.default_timer()

print('Time: ', stop - start)

# low water level
inv_inlow = np.arange(0,33*2, 2)
s_ind_low = np.arange(1,33*2, 2)
final_inv_low = np.sum(mapping_space[:,inv_ind_low],axis = 1)/33
final_s_low = np.sum(mapping_space[:,s_ind_low], axis = 1)/33
invAs = np.array((final_inv_low, final_s_low))
invAs = invAs.T
np.savetxt('low_new_6_stages.csv',  invAs, delimiter =", ",  fmt ='% s')

# medium water level
inv_ind_medium = np.arange(33*2,66*2, 2)
s_ind_medium = np.arange(33*2+1,66*2, 2)
final_inv_medium = np.sum(mapping_space[:,inv_ind_medium],axis = 1)/33
final_s_medium = np.sum(mapping_space[:,s_ind_medium], axis = 1)/33
invAs = np.array((final_inv_medium, final_s_medium))
invAs = invAs.T
np.savetxt('medium_new_6_stages.csv',  invAs, delimiter =", ",  fmt ='% s')

# high water level
inv_ind_high = np.arange(66*2,100*2, 2)
s_ind_high = np.arange(66*2+1,100*2, 2)
final_inv_high = np.sum(mapping_space[:,inv_ind_high],axis = 1)/34
final_s_high = np.sum(mapping_space[:,s_ind_high], axis = 1)/34
invAs = np.array((final_inv_high, final_s_high))
invAs = invAs.T
np.savetxt('high_new_6_stages.csv',  invAs, delimiter =", ",  fmt ='% s')
