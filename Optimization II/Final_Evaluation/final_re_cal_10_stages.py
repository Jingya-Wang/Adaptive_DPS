#!/usr/bin/env python
# coding: utf-8


'''
DPS project;
time series decomposition;
SLR case
Final evaluation

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
#print(one_water_level)

# read meta-policies: tagged with seeds; running 10 seeds here

mp_1 = pd.read_csv('mp_pred_10_stages_0.csv', header = None)
mp_1 = mp_1.to_numpy()

mp_2 = pd.read_csv('mp_pred_10_stages_19.csv', header = None)
mp_2 = mp_2.to_numpy()

mp_3 = pd.read_csv('mp_pred_10_stages_29.csv', header = None)
mp_3 = mp_3.to_numpy()

mp_4 = pd.read_csv('mp_pred_10_stages_39.csv', header = None)
mp_4 = mp_4.to_numpy()

mp_5 = pd.read_csv('mp_pred_10_stages_42.csv', header = None)
mp_5 = mp_5.to_numpy()

mp_6 = pd.read_csv('mp_pred_10_stages_49.csv', header = None)
mp_6 = mp_6.to_numpy()

mp_7 = pd.read_csv('mp_pred_10_stages_59.csv', header = None)
mp_7 = mp_7.to_numpy()

mp_8 = pd.read_csv('mp_pred_10_stages_69.csv', header = None)
mp_8 = mp_8.to_numpy()

mp_9 = pd.read_csv('mp_pred_10_stages_119.csv', header = None)
mp_9 = mp_9.to_numpy()

mp_10 = pd.read_csv('mp_pred_10_stages_159.csv', header = None)
mp_10 = mp_10.to_numpy()

mp_fake = np.concatenate((mp_1, mp_2, mp_3, mp_4, mp_5, mp_6, mp_7, mp_8, mp_9, mp_10),axis = 0)

mp_dup = [tuple(row) for row in mp_fake]
mp = np.unique(mp_dup, axis = 0)


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
base_year = 2015
sl2000 = 0.0
time_step = 100
time_stage = int(n_years/time_step)
n_obj = 2


##BH_FH
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
def pickup_strategy(probs, beliefs_space, time_stage_ind):
    distances = np.zeros(n_beliefs)
    if time_stage_ind == 0:
        strategy_name = 1
    else:
        for i in range(n_beliefs):
            distances[i] = sum([abs(j-k) for j,k in zip(probs,beliefs_space[i,:])])

        ind = int(np.where(distances == min(distances))[0]) + 1

        strategy_name = int(ind)

    return strategy_name
    


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
    #the decision for first time stage is just the combination 1
    timestage1_name = pickup_strategy(probs[:,0],beliefs_space, 0)
    #the decision for second time stage is based on the first prediction,
    timestage2_name = pickup_strategy(probs[:,0], beliefs_space, 1)
    #the decicion for third time stage is based on the second prediction (the third prediction actually doesn't matter)
    timestage3_name = pickup_strategy(probs[:,1], beliefs_space, 2)
    timestage4_name = pickup_strategy(probs[:,2], beliefs_space, 3)
    timestage5_name = pickup_strategy(probs[:,3], beliefs_space, 4)
    timestage6_name = pickup_strategy(probs[:,4], beliefs_space, 5)
    timestage7_name = pickup_strategy(probs[:,5], beliefs_space, 6)
    timestage8_name = pickup_strategy(probs[:,6], beliefs_space, 7)
    timestage9_name = pickup_strategy(probs[:,7], beliefs_space, 8)
    timestage10_name = pickup_strategy(probs[:,8], beliefs_space, 9)
    return timestage1_name, timestage2_name, timestage3_name, timestage4_name, timestage5_name, timestage6_name, timestage7_name, timestage8_name, timestage9_name, timestage10_name


mp_mapping_space = np.zeros((len(mp),len(water_level) * n_obj))

mp_mappings = np.zeros((len(mp), n_vars * n_beliefs))



def total_inv_s(timestage1, timestage2, timestage3, timestage4, timestage5, timestage6, timestage7,\
                timestage8, timestage9, timestage10,one_water_level):
    inv = np.zeros(n_years)
    s = np.zeros(n_years)
    for t in range(0,time_step):
        rbf_vars = timestage1
        x = rbf_vars[0::3]
        r = rbf_vars[1::3]
        w = rbf_vars[2::3]
        if t == 0:
            prev_h = h_0
        u = calc_heightening(t, prev_h, one_water_level, n, year_offset, x, r, w, h_0)
        inv[t], s[t], current_h, flood_rel = calc_inv_loss(t, prev_h, u, one_water_level, year_offset, V_0, gamma, zeta, h_0, kappa, upsilon, lam)
        prev_h = current_h
    for t in range(time_step,time_step*2):
        rbf_vars = timestage2
        x = rbf_vars[0::3]
        r = rbf_vars[1::3]
        w = rbf_vars[2::3]
        u = calc_heightening(t, prev_h, one_water_level, n, year_offset, x, r, w, h_0)
        inv[t], s[t], current_h, flood_rel = calc_inv_loss(t, prev_h, u, one_water_level, year_offset, V_0, gamma, zeta, h_0, kappa, upsilon, lam)
        #current_h = prev_h + u[count,t]
        prev_h = current_h
    for t in range(time_step*2,time_step*3):
        rbf_vars = timestage3
        x = rbf_vars[0::3]
        r = rbf_vars[1::3]
        w = rbf_vars[2::3]
        u = calc_heightening(t, prev_h, one_water_level, n, year_offset, x, r, w, h_0)
        inv[t], s[t], current_h, flood_rel = calc_inv_loss(t, prev_h, u, one_water_level, year_offset, V_0, gamma, zeta, h_0, kappa, upsilon, lam)
        #current_h = prev_h + u[count,t]
        prev_h = current_h
    for t in range(time_step*3,time_step*4):
        rbf_vars = timestage4
        x = rbf_vars[0::3]
        r = rbf_vars[1::3]
        w = rbf_vars[2::3]
        u = calc_heightening(t, prev_h, one_water_level, n, year_offset, x, r, w, h_0)
        inv[t], s[t], current_h, flood_rel = calc_inv_loss(t, prev_h, u, one_water_level, year_offset, V_0, gamma, zeta, h_0, kappa, upsilon, lam)
        #current_h = prev_h + u[count,t]
        prev_h = current_h
    for t in range(time_step*4,time_step*5):
        rbf_vars = timestage5
        x = rbf_vars[0::3]
        r = rbf_vars[1::3]
        w = rbf_vars[2::3]
        u = calc_heightening(t, prev_h, one_water_level, n, year_offset, x, r, w, h_0)
        inv[t], s[t], current_h, flood_rel = calc_inv_loss(t, prev_h, u, one_water_level, year_offset, V_0, gamma, zeta, h_0, kappa, upsilon, lam)
        #current_h = prev_h + u[count,t]
        prev_h = current_h
    for t in range(time_step*5,time_step*6):
        rbf_vars = timestage6
        x = rbf_vars[0::3]
        r = rbf_vars[1::3]
        w = rbf_vars[2::3]
        u = calc_heightening(t, prev_h, one_water_level, n, year_offset, x, r, w, h_0)
        inv[t], s[t], current_h, flood_rel = calc_inv_loss(t, prev_h, u, one_water_level, year_offset, V_0, gamma, zeta, h_0, kappa, upsilon, lam)
        #current_h = prev_h + u[count,t]
        prev_h = current_h
    for t in range(time_step*6,time_step*7):
        rbf_vars = timestage7
        x = rbf_vars[0::3]
        r = rbf_vars[1::3]
        w = rbf_vars[2::3]
        u = calc_heightening(t, prev_h, one_water_level, n, year_offset, x, r, w, h_0)
        inv[t], s[t], current_h, flood_rel = calc_inv_loss(t, prev_h, u, one_water_level, year_offset, V_0, gamma, zeta, h_0, kappa, upsilon, lam)
        #current_h = prev_h + u[count,t]
        prev_h = current_h
    for t in range(time_step*7,time_step*8):
        rbf_vars = timestage8
        x = rbf_vars[0::3]
        r = rbf_vars[1::3]
        w = rbf_vars[2::3]
        u = calc_heightening(t, prev_h, one_water_level, n, year_offset, x, r, w, h_0)
        inv[t], s[t], current_h, flood_rel = calc_inv_loss(t, prev_h, u, one_water_level, year_offset, V_0, gamma, zeta, h_0, kappa, upsilon, lam)
        #current_h = prev_h + u[count,t]
        prev_h = current_h
    for t in range(time_step*8,time_step*9):
        rbf_vars = timestage9
        x = rbf_vars[0::3]
        r = rbf_vars[1::3]
        w = rbf_vars[2::3]
        u = calc_heightening(t, prev_h, one_water_level, n, year_offset, x, r, w, h_0)
        inv[t], s[t], current_h, flood_rel = calc_inv_loss(t, prev_h, u, one_water_level, year_offset, V_0, gamma, zeta, h_0, kappa, upsilon, lam)
        #current_h = prev_h + u[count,t]
        prev_h = current_h
    for t in range(time_step*9,time_step*10):
        rbf_vars = timestage10
        x = rbf_vars[0::3]
        r = rbf_vars[1::3]
        w = rbf_vars[2::3]
        u = calc_heightening(t, prev_h, one_water_level, n, year_offset, x, r, w, h_0)
        inv[t], s[t], current_h, flood_rel = calc_inv_loss(t, prev_h, u, one_water_level, year_offset, V_0, gamma, zeta, h_0, kappa, upsilon, lam)
        #current_h = prev_h + u[count,t]
        prev_h = current_h

    total_inv, total_v = discounted(inv, s, n_years, delta, delta_1)

    return total_inv, total_v



def mapping(sow, one_water_level,timestage1_name,timestage2_name,timestage3_name, timestage4_name, timestage5_name, timestage6_name, timestage7_name, timestage8_name, timestage9_name, timestage10_name):
    print(timestage1_name, timestage2_name, timestage3_name, timestage4_name, timestage5_name, timestage6_name, timestage7_name, timestage8_name, timestage9_name, timestage10_name)
  
    for i in range(len(mp)):
        total_inv, total_s = total_inv_s(mp[i,0:12], mp[i,(timestage2_name-1)*12 : (timestage2_name-1)*12+12],\
                                         mp[i,(timestage3_name-1)*12 : (timestage3_name-1)*12+12], \
                                         mp[i,(timestage4_name-1)*12 : (timestage4_name-1)*12+12], \
                                         mp[i,(timestage5_name-1)*12 : (timestage5_name-1)*12+12], \
                                         mp[i,(timestage6_name-1)*12 : (timestage6_name-1)*12+12], \
                                         mp[i,(timestage7_name-1)*12 : (timestage7_name-1)*12+12], \
                                         mp[i,(timestage8_name-1)*12 : (timestage8_name-1)*12+12], \
                                         mp[i,(timestage9_name-1)*12 : (timestage9_name-1)*12+12], \
                                         mp[i,(timestage10_name-1)*12 : (timestage10_name-1)*12+12], \
                                         one_water_level)

        mp_mapping_space[i, sow * n_obj] = total_inv
        mp_mapping_space[i, sow * n_obj + 1] = total_s


    return mp_mapping_space



for i in range(len(water_level)):
    one_water_level = water_level[i,:]
    new_water_level = one_water_level[n:]
    timestage1_name, timestage2_name, timestage3_name, timestage4_name, timestage5_name, timestage6_name, \
    timestage7_name, timestage8_name, timestage9_name, timestage10_name = selections(time_stage, years, time_step, new_water_level, n_years, bin_matrix)
    mp_mapping_space = mapping(i, one_water_level,timestage1_name,timestage2_name,timestage3_name, timestage4_name, timestage5_name, timestage6_name, \
                           timestage7_name, timestage8_name, timestage9_name, timestage10_name)

inv_ind = np.arange(0,len(water_level) * n_obj, 2)
s_ind = np.arange(1,len(water_level) * n_obj, 2)
final_inv = np.sum(mp_mapping_space[:,inv_ind],axis = 1)/len(water_level)
final_s = np.sum(mp_mapping_space[:,s_ind], axis = 1)/len(water_level)

                           
                           
invAs = np.array((final_inv, final_s))
invAs = invAs.T

stop = timeit.default_timer()

print('Time: ', stop - start)


np.savetxt('final_10_stages.csv',  invAs, delimiter =", ",  fmt ='% s')


# low water level
inv_ind = np.arange(0,33*2, 2)
s_ind = np.arange(1,33*2, 2)
final_inv = np.sum(mp_mapping_space[:,inv_ind],axis = 1)/33
final_s = np.sum(mp_mapping_space[:,s_ind], axis = 1)/33

invAs = np.array((final_inv, final_s))
invAs = invAs.T
np.savetxt('low_final_10_stages.csv',  invAs, delimiter =", ",  fmt ='% s')


# medium water level
inv_ind = np.arange(33*2,66*2, 2)
s_ind = np.arange(33*2+1,66*2, 2)
final_inv = np.sum(mp_mapping_space[:,inv_ind],axis = 1)/33
final_s = np.sum(mp_mapping_space[:,s_ind], axis = 1)/33
invAs = np.array((final_inv, final_s))
invAs = invAs.T
np.savetxt('medium_final_10_stages.csv',  invAs, delimiter =", ",  fmt ='% s')


# high water level
inv_ind = np.arange(66*2,100*2, 2)
s_ind = np.arange(66*2+1,100*2, 2)
final_inv = np.sum(mp_mapping_space[:,inv_ind],axis = 1)/34
final_s = np.sum(mp_mapping_space[:,s_ind], axis = 1)/34
invAs = np.array((final_inv, final_s))
invAs = invAs.T
np.savetxt('high_final_10_stages.csv',  invAs, delimiter =", ",  fmt ='% s')




