#!/usr/bin/env python
# coding: utf-8



'''
DPS project;
time series decomposition;
SLR case

'''

import numpy as np
import pandas as pd
import math
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from borg import *


import timeit
start = timeit.default_timer()

# the calculation for water level is commented out for saving time; it has been run before running this script
#dd = '../DPS/DataCommonsArchive/ext_data/sow_a_b_c_tstar_cstar_loc_scale_shape_100k.txt'
#coefs = np.loadtxt(dd, delimiter=" ", unpack=True)

dd = '../data/sampled_water_level_6_2.csv'
water_level = pd.read_csv(dd, header = None)
water_level = water_level.to_numpy()




####################################
##        assign values       ##
####################################
n = 30
year_offset = 30
t_years = 330
n_years = 300
years = np.arange(0,n_years,1)
base_year = 2015
sl2000 = 0.0
time_step = 100
time_stage = int(n_years/time_step)

##BH_FH
n_rbf = 4
n_vars = n_rbf * 3

## parameters
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
n_sow = 10000
#n_sow = 100
inv_n_sow = 1/n_sow

# constraint
rel_threshold = 0.8




# ####################################
# ##        initialize       ##
# ####################################
# sl = np.zeros(t_years)
# sl_centered = np.zeros(t_years)
# surge = np.zeros(t_years)
# water_level_orig = np.zeros(t_years)
# water_level = np.zeros(t_years)





# # read in the coefficient file
# coefs = np.loadtxt('../DPS/DataCommonsArchive/ext_data/sow_a_b_c_tstar_cstar_loc_scale_shape_100k.txt', delimiter=" ", unpack=True)

# data = coefs[:,1]

# slr_a = data[0]
# slr_b = data[1]
# slr_c = data[2]
# t_star = data[3]- base_year
# #t_star = 10
# c_star = data[4]

# mu = data[5]
# sigma = data[6]
# xi = data[7]


# def slr(slr_a, slr_b, slr_c, t_star, c_star, t):
#     if t <= t_star:
#         z_t = slr_a + slr_b*t + slr_c*t*t
#     else:
#         z_t = slr_a + slr_b*t + slr_c*t*t + c_star*(t-t_star)
#     return z_t

# def storm_surge(mu,sigma,xi,p):
#     # mu is location parameter of the GEV function
#     # sigma is the scale parameter of the GEV fuction
#     # xi is he shape parameter of the GEV function
#     # p is a quantile randomly sample from a uniform distribution from zero to on inclusive
#     if xi ==0:
#         x_t = mu + sigma*np.log(1/(np.log(1/p)))
#     else:
#         x_t = mu + sigma*(math.pow(np.log(1/p),(-1)*xi)-1)/xi
#     return x_t





# ####################################
# ##        Calculate the slr       ##
# ####################################

# for i in range(0,t_years):
#     # Calculate the year of simulation for the SLR model
#     # SL starts in 1970, base year in SL model is 2015
#     calc_year = i - year_offset - 15
#     if calc_year > 0:
#         sign_year = 1
#     elif calc_year < 0:
#         sign_year = -1
#     else:
#         sign_year = 0

#     # Update the SLR
#     sl[i] = slr(slr_a,slr_b,slr_c,t_star,c_star,i)
    
    
#     # generate storm surge
#     p = np.random.uniform()
#     surge[i] = storm_surge(mu,sigma,xi,p)
    
#     water_level_orig[i] = sl[i] + surge[i]
    
# # uncentered water level
# new_water_level_orig = water_level_orig[n:]
    
# # Center the SLR on the year 2000
# sl2000 = sl[30]
# for i in range(0,t_years):
#     sl_centered[i] = sl[i] - sl2000

#     water_level[i] = sl_centered[i] + surge[i]
# new_water_level = water_level[n:] # first 30-year data is erased




####################################
##        Calculate the slr       ##
####################################
## will generate this before running this problem

def calc_slr(sow_ind, coefs):
    
    seed = sow_ind * 13 + 1
    sl = np.zeros(t_years)
    sl_centered = np.zeros(t_years)
    surge = np.zeros(t_years)
    #water_level_orig = np.zeros(t_years)
    water_level = np.zeros(t_years)

    # read in the coefficient file
    

    data = coefs[:,sow_ind]

    slr_a = data[0]
    slr_b = data[1]
    slr_c = data[2]
    t_star = data[3]- base_year
    #t_star = 10
    c_star = data[4]

    mu = data[5]
    sigma = data[6]
    xi = data[7]


    def slr(slr_a, slr_b, slr_c, t_star, c_star, t):
        if t <= t_star:
            z_t = slr_a + slr_b*t + slr_c*t*t
        else:
            z_t = slr_a + slr_b*t + slr_c*t*t + c_star*(t-t_star)
        return z_t

    def storm_surge(mu,sigma,xi,p):
        # mu is location parameter of the GEV function
        # sigma is the scale parameter of the GEV fuction
        # xi is he shape parameter of the GEV function
        # p is a quantile randomly sample from a uniform distribution from zero to on inclusive
        if xi ==0:
            x_t = mu + sigma*np.log(1/(np.log(1/p)))
        else:
            x_t = mu + sigma*(math.pow(np.log(1/p),(-1)*xi)-1)/xi
        return x_t
    
    for i in range(0,t_years):
    # Calculate the year of simulation for the SLR model
    # SL starts in 1970, base year in SL model is 2015
        calc_year = i - year_offset - 15
        if calc_year > 0:
            sign_year = 1
        elif calc_year < 0:
            sign_year = -1
        else:
            sign_year = 0

        # Update the SLR
        #sl[i] = slr(slr_a,slr_b,slr_c,t_star,c_star,i)
        sl[i] = slr_a + slr_b * calc_year + sign_year * slr_c * (calc_year*calc_year)
        if calc_year >= t_star:
            sl[i] = sl[i] + c_star * (calc_year - t_star)

        # generate storm surge
        np.random.seed(seed)
        p = np.random.uniform()
        surge[i] = storm_surge(mu,sigma,xi,p)

        #water_level_orig[i] = sl[i] + surge[i]

        # uncentered water level
        #new_water_level_orig = water_level_orig[n:]

    # Center the SLR on the year 2000
    sl2000 = sl[30]
    for i in range(0,t_years):
        sl_centered[i] = sl[i] - sl2000

        water_level[i] = sl_centered[i] + surge[i]
    #new_water_level = water_level[n:] # first 30-year data is erased
    
    water_level = water_level * 0.1
    return water_level



####################################
## get dike heightening for time t
## based on calculating state varaibles, \
## BH and FH, and then heightening
####################################

def calc_heightening(t, prev_h, water_level, n, year_offset, x, r, w, h_0):
    
    # set the previous height as h_0; otherwise, set it as the previous timestep
    prev_sl = water_level[t-1+year_offset]
#     prev_h = h_0
#     if t > 0:
#         prev_h = prev_h
    
    ####################################
    ## fit the simple linear regression based on
    ## past n-year water level data. The coefficients
    ## are used as state variables to describe states##
    ####################################
    
    sx = 0
    sy = 0
    sxx = 0
    sxy = 0
    ssr = 0
    for i in range(n):
        ind = t + year_offset - n + i
        sx += i
        sy += water_level[ind]
        sxx += i * i
        sxy += i * water_level[ind]
    slope = (sxy - (sx * sy)/n) / (sxx - (sx * sx)/n)
    intercept = (sy/n) - slope * (sx/n)
    for i in range(n):
        obs = water_level[t+year_offset-n+i]
        yfit = intercept + slope*i
        ssr += (yfit-obs)*(yfit-obs)
    mean_slr_rate = slope
    srss = math.sqrt(ssr)
    #obs = np.zeros(n)
    #for i in range(n):
    #    obs[i] = water_level[t+year_offset-n+i]
    #mean_slr_rate, srss = state_variables(obs,n)

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

    return FH_t, BH_t, u_t




####################################
## calculate expected loss and investment
## at time t
####################################
def calc_inv_loss(t, prev_h, u_t, water_level, year_offset, V_0, gamma, zeta, h_0, kappa, upsilon, lam):
    
#     prev_h = h_0
    
#     if t > 0:
#         prev_h = prev_h
    flood_rel = 0
    current_h = prev_h + u_t

    # expected loss
    V_t = V_0 * math.exp(gamma * t) * math.exp(zeta * (current_h - h_0))
    
    if current_h <= water_level[t + year_offset]:
        s_t = V_t
        flood_rel = flood_rel + 1
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
            
        FH[i], BH[i], u[i] = calc_heightening(i, prev_h, water_level, n, year_offset, x, r, w, h_0)
        inv[i], s[i], current_h, flood_rel[i] = calc_inv_loss(i, prev_h, u[i], water_level, year_offset, V_0, gamma, zeta, h_0, kappa, upsilon, lam)
        prev_h = current_h
    total_inv, total_s = discounted(inv, s, n_years, delta, delta_1)
    
    return total_inv, total_s, flood_rel




####################################
## problem formulation, get ready to be optimized
####################################
def slr_problem(*rbf_vars):
    ####################################
    ## wrap up
    ## generate all SOWs
    ####################################
    #np.random.seed(0)
    total_inv = np.zeros(n_sow)
    total_s = np.zeros(n_sow)
    total_flood_rel = np.zeros((n_years, n_sow))
    
    x = rbf_vars[0::3]
    r = rbf_vars[1::3]
    #orig_w = rbf_vars[2::3]
    w = rbf_vars[2::3]
    #w = np.zeros(len(orig_w))

    objs = [0.0] * nobjs
    constrs = [0.0] * nconstrs
    
    ###############
    ## if use kernel rbf, need to normalize
    ###############
    ##Normalize weights to sum to 1
    #total = sum(orig_w)
    #if total != 0.0:
    #    for i in range(len(orig_w)):
    #        w[i] = orig_w[i] / total
    #else:
    #    for i in range(len(w)):
    #        w[i] = 1 / n_rbf

    for i in range(n_sow):
        total_inv[i], total_s[i], total_flood_rel[:,i] = calc_model(n_years, h_0, water_level[i,:], n, year_offset, x, r, w, V_0, gamma, zeta, kappa, upsilon, lam, delta, delta_1)

    # assign objectives weights
    
    sum_inv_1 = sum(total_inv[0:5000])
    sum_inv_2 = sum(total_inv[5000:10000])

    sum_s_1 = sum(total_s[0:5000])
    sum_s_2 = sum(total_s[5000:10000])

    weighted_inv = (sum_inv_1 * 0.7 + sum_inv_2 * 0.3) * inv_n_sow
    weighted_s = (sum_s_1 * 0.7 + sum_s_2 * 0.3) * inv_n_sow

    objs[0] = weighted_inv
    objs[1] = weighted_s
    
    sum_flood_rel = np.sum(total_flood_rel, axis = 1)
    
    temp_rel = 0
    rel_constraint = 0
    for i in range(n_years):
        temp_rel = 1 - (sum_flood_rel[i] * inv_n_sow)
        if temp_rel < rel_threshold:
            rel_constraint = rel_constraint + 1
            

    constrs[0] = rel_constraint
    return objs, constrs
    


### run with borg
nvars = 12
nobjs = 2
nconstrs = 1
Configuration.startMPI()

borg = Borg(nvars, nobjs, nconstrs, slr_problem)

# set bounds and epsilons for the Lake problem
borg.setBounds(*[[0, 300], [-50, 50], [-5, 5]] * int((nvars / 3)))
borg.setEpsilons(0.2, 0.2)

#maxtime = 3.0
num_func_evals = 200000
#runtime_freq = 200
#result = borg.solve()
#runtime_filename = main_output_file_dir + os_fold + 'runtime_file_seed_' + str(j+1) + '.runtime'
#runtime_filename = 'runtime_c1.runtime'
result = borg.solveMPI(maxEvaluations = num_func_evals)
Configuration.stopMPI()

#result = borg.solve({"maxEvaluations":1000})

objectives_total = np.empty(shape=[0,nobjs])
strategies_total = np.empty(shape=[0,nvars])

if result:
    for solution in result:
        objectives = solution.getObjectives()
        objectives = np.column_stack(objectives)
        objectives_total = np.append(objectives_total,objectives,axis=0)
        strategies = solution.getVariables()
        strategies = np.column_stack(strategies)
        strategies_total = np.append(strategies_total,strategies,axis=0)

if result:
    np.savetxt("objectives_c2.csv", objectives_total, delimiter=",")
    np.savetxt("strategies_c2.csv", strategies_total, delimiter=",")

print(objectives_total)
print(strategies_total)

stop = timeit.default_timer()

print('Time: ', stop - start)

