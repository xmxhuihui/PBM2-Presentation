#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 18:59:23 2020

@author: xmxhuihui
"""


import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import seaborn as sns
# Number of excitatory and inhibitory neurons
N_E = 80
N_I = 20
n_neurons = N_E + N_I
n_sessions = 6
total_time = 5000

# All the parameters from Supplementary table from the paper.
W_EI = 0.44
W_IE = 0.66
W_II = 0.54
W_EE = 0.37
W_EI2 = 0.49
W_IE2 = 0.65
W_II2 = 0.53
W_EE2 = 0.26

mu_EI = W_EI
mu_IE = W_IE
mu_II = W_II
sigma_EI2 = W_EI2 - W_EI ** 2
sigma_IE2 = W_IE2 - W_IE2 ** 2
sigma_II2 = W_II2 - W_II ** 2
sigma_EI = np.sqrt(sigma_EI2)
sigma_IE = np.sqrt(sigma_IE2)
sigma_II = np.sqrt(sigma_II2)
#print(str(sigma_EI2) + '\n' + str(sigma_IE2) + '\n' + str(sigma_II2))

theta = 33
tau_m = 10
H_E = 77.6
H_I = 57.8
v_R = 24.75
spike = 150


# Extracting E->E connectivity from the spine imaging data
c_EE = 0.2
path = "Global_Spines_info.csv"
spines_info = pd.read_csv(path)
spines_info.drop('Unnamed: 0', axis=1, inplace=True)

spines_IS1 = spines_info.loc[spines_info['Starting Imaging Session'] == 1]
# spines_IS1.head(100)
S = spines_IS1['Volume'].mean()
g = W_EE / S
#print(g)


# Connectivity matrix 8*8
# EI
# I*
def W_Construction():
    c_EE = 0.2
    c_EI = 0.4
    c_IE = 0.3
    c_II = 0.4
    W = np.zeros((n_neurons, n_neurons))

    # E-I connections
    for i in range(N_E):
        for j in range(N_E, n_neurons):
            if random.uniform(0, 1) <= c_EI:
                W[i, j] = -np.random.lognormal(mu_EI, sigma_EI)

    # I-E connections
    for i in range(N_E, n_neurons):
        for j in range(N_E):
            if random.uniform(0, 1) <= c_IE:
                W[i, j] = np.random.lognormal(mu_IE, sigma_IE)

    # I-I connections
    for i in range(N_E, n_neurons):
        for j in range(N_E, n_neurons):
            if random.uniform(0, 1) <= c_II:
                W[i, j] = -np.random.lognormal(mu_II, sigma_II)

    # E-E connections
    #index_list = spines_info['Global_SpineID'].loc[(spines_info['Starting Imaging Session']<=IS)&(spines_info['Ending Imaging Session']>=IS)]
    #print(len(index_list))
    for i in range(N_E):
        for j in range(N_E):
            if random.uniform(0, 1) <= c_EE:
                index = random.randint(1,3688)
                W[i, j] = spines_info['Volume'].loc[spines_info['Global_SpineID'] == index].values[0] * g
            else:
                W[i, j] = 0
    return W

W = W_Construction()


v_original = np.zeros((n_neurons, total_time))
h = np.zeros((n_neurons, total_time))
r = np.zeros(n_neurons)
e_firing_time=[[] for i in range(N_E)]
i_firing_time=[[] for i in range(N_I)]
e_firing_rate=[]
i_firing_rate=[]
# Recording the state of each neuron in the last timestep
for i in range(n_neurons):
    v_original[i, 0] = v_R
t = range(total_time - 1)
    # For excitatory neurons
for dt in t:

    for i in range(N_E):
        for j in range(n_neurons):
            h[i, dt] = h[i, dt] + W[i, j] * r[j]
        if v_original[i, dt] == spike:
            v_original[i, dt + 1] = v_R
        else:
            v_original[i, dt + 1] = v_original[i, dt] - v_original[i, dt] / tau_m + h[i, dt] + H_E / tau_m
            if v_original[i, dt + 1] >= theta:
                v_original[i, dt + 1] = spike
                r[i] = 1
                e_firing_time[i].append(dt+1)
                #e_rates[IS-1,i]=e_rates[IS-1,i]+1
            else:
                r[i] = 0

    # firing_rate = e_spikes / total_time * 1000
    # e_firing_rates[IS-1,i]=firing_rate
# For inhibitory neurons
    for i in range(N_E, n_neurons):
        for j in range(n_neurons):
            h[i, dt] = h[i, dt] + W[i, j] * r[j]
        if v_original[i, dt] == spike:
            v_original[i, dt + 1] = v_R
        else:
            v_original[i, dt + 1] = v_original[i, dt] - v_original[i, dt] / tau_m + h[i, dt] + H_I / tau_m
            if v_original[i, dt + 1] >= theta:
                v_original[i, dt + 1] = spike
                r[i] = 1
                i_firing_time[i-N_E].append(dt+1)
                # i_rates[IS-1,i-N_E]=i_rates[IS-1,i-N_E]+1
            else:
                r[i] = 0
for i in range(n_neurons):
    if i < N_E:
        if len(e_firing_time[i])==1:
            e_firing_rate.append(1)
        else:
            e_firing_rate.append((e_firing_time[i][-1]-e_firing_time[i][0])/len(e_firing_time[i]))
    else:
        if len(i_firing_time[i-N_E])==1:
            i_firing_rate.append(1)
        else:
            i_firing_rate.append((i_firing_time[i-N_E][-1]-i_firing_time[i-N_E][0])/len(i_firing_time[i-N_E]))
print(e_firing_rate)
print(i_firing_rate)

# Rewiring E-E
W_rEE=W.copy()
for i in range(N_E):
    for j in range(N_E):
        index = random.randint(1,3688)
        W_rEE[i, j] = spines_info['Volume'].loc[spines_info['Global_SpineID'] == index].values[0] * g
v = np.zeros((n_neurons, total_time))
h = np.zeros((n_neurons, total_time))
r = np.zeros(n_neurons)
e_firing_time_EE=[[] for i in range(N_E)]
i_firing_time_EE=[[] for i in range(N_I)]
e_firing_rate_EE=[]
i_firing_rate_EE=[]
for i in range(n_neurons):
    v[i, 0] = v_R
t = range(total_time - 1)
    # For excitatory neurons
for dt in t:

    for i in range(N_E):
        for j in range(n_neurons):
            h[i, dt] = h[i, dt] + W_rEE[i, j] * r[j]
        if v[i, dt] == spike:
            v[i, dt + 1] = v_R
        else:
            v[i, dt + 1] = v[i, dt] - v[i, dt] / tau_m + h[i, dt] + H_E / tau_m
            if v[i, dt + 1] >= theta:
                v[i, dt + 1] = spike
                r[i] = 1
                e_firing_time_EE[i].append(dt+1)
                #e_rates[IS-1,i]=e_rates[IS-1,i]+1
            else:
                r[i] = 0

    # firing_rate = e_spikes / total_time * 1000
    # e_firing_rates[IS-1,i]=firing_rate
# For inhibitory neurons
    for i in range(N_E, n_neurons):
        for j in range(n_neurons):
            h[i, dt] = h[i, dt] + W_rEE[i, j] * r[j]
        if v[i, dt] == spike:
            v[i, dt + 1] = v_R
        else:
            v[i, dt + 1] = v[i, dt] - v[i, dt] / tau_m + h[i, dt] + H_I / tau_m
            if v[i, dt + 1] >= theta:
                v[i, dt + 1] = spike
                r[i] = 1
                i_firing_time_EE[i-N_E].append(dt+1)
                # i_rates[IS-1,i-N_E]=i_rates[IS-1,i-N_E]+1
            else:
                r[i] = 0
for i in range(n_neurons):
    if i < N_E:
        if len(e_firing_time_EE[i])==1:
            e_firing_rate_EE.append(1)
        else:
            e_firing_rate_EE.append((e_firing_time_EE[i][-1]-e_firing_time_EE[i][0])/len(e_firing_time_EE[i]))
    else:
        if len(i_firing_time_EE[i-N_E])==1:
            i_firing_rate_EE.append(1)
        else:
            i_firing_rate_EE.append((i_firing_time_EE[i-N_E][-1]-i_firing_time_EE[i-N_E][0])/len(i_firing_time_EE[i-N_E]))
print(e_firing_rate_EE)
print(i_firing_rate_EE)
e_firing_rate_mean=np.array(e_firing_rate).mean()
e_firing_rate_EE_mean=np.array(e_firing_rate_EE).mean()
tmp=0
for i in range(N_E):
    tmp+=(e_firing_rate[i]-e_firing_rate_mean)*(e_firing_rate_EE[i]-e_firing_rate_EE_mean)
tmp1=0
tmp2=0
for i in range(N_E):
    tmp1+=(e_firing_rate[i]-e_firing_rate_mean)**2
    tmp2+=(e_firing_rate_EE[i]-e_firing_rate_EE_mean)**2
corr_coef_EE_E=tmp/(np.sqrt(tmp1)*np.sqrt(tmp2))
print(corr_coef_EE_E)

i_firing_rate_mean=np.array(i_firing_rate).mean()
i_firing_rate_EE_mean=np.array(i_firing_rate_EE).mean()
tmp=0
for i in range(N_I):
    tmp+=(i_firing_rate[i]-i_firing_rate_mean)*(i_firing_rate_EE[i]-i_firing_rate_EE_mean)
tmp1=0
tmp2=0
for i in range(N_I):
    tmp1+=(i_firing_rate[i]-i_firing_rate_mean)**2
    tmp2+=(i_firing_rate_EE[i]-i_firing_rate_EE_mean)**2
corr_coef_EE_I=tmp/(np.sqrt(tmp1)*np.sqrt(tmp2))
print(corr_coef_EE_I)

# Rewiring E_I
W_rEI=W.copy()
for i in range(N_E):
    for j in range(N_E, n_neurons):
        W[i, j] = -np.random.lognormal(mu_EI, sigma_EI)
v = np.zeros((n_neurons, total_time))
h = np.zeros((n_neurons, total_time))
r = np.zeros(n_neurons)
e_firing_time_EI=[[] for i in range(N_E)]
i_firing_time_EI=[[] for i in range(N_I)]
e_firing_rate_EI=[]
i_firing_rate_EI=[]
for i in range(n_neurons):
    v[i, 0] = v_R
t = range(total_time - 1)
    # For excitatory neurons
for dt in t:

    for i in range(N_E):
        for j in range(n_neurons):
            h[i, dt] = h[i, dt] + W_rEI[i, j] * r[j]
        if v[i, dt] == spike:
            v[i, dt + 1] = v_R
        else:
            v[i, dt + 1] = v[i, dt] - v[i, dt] / tau_m + h[i, dt] + H_E / tau_m
            if v[i, dt + 1] >= theta:
                v[i, dt + 1] = spike
                r[i] = 1
                e_firing_time_EI[i].append(dt+1)
                #e_rates[IS-1,i]=e_rates[IS-1,i]+1
            else:
                r[i] = 0

    # firing_rate = e_spikes / total_time * 1000
    # e_firing_rates[IS-1,i]=firing_rate
# For inhibitory neurons
    for i in range(N_E, n_neurons):
        for j in range(n_neurons):
            h[i, dt] = h[i, dt] + W_rEI[i, j] * r[j]
        if v[i, dt] == spike:
            v[i, dt + 1] = v_R
        else:
            v[i, dt + 1] = v[i, dt] - v[i, dt] / tau_m + h[i, dt] + H_I / tau_m
            if v[i, dt + 1] >= theta:
                v[i, dt + 1] = spike
                r[i] = 1
                i_firing_time_EI[i-N_E].append(dt+1)
                # i_rates[IS-1,i-N_E]=i_rates[IS-1,i-N_E]+1
            else:
                r[i] = 0
for i in range(n_neurons):
    if i < N_E:
        if len(e_firing_time_EI[i])==1:
            e_firing_rate_EI.append(1)
        else:
            e_firing_rate_EI.append((e_firing_time_EI[i][-1]-e_firing_time_EI[i][0])/len(e_firing_time_EI[i]))
    else:
        if len(i_firing_time_EI[i-N_E])==1:
            i_firing_rate_EI.append(1)
        else:
            i_firing_rate_EI.append((i_firing_time_EI[i-N_E][-1]-i_firing_time_EI[i-N_E][0])/len(i_firing_time_EI[i-N_E]))
print(e_firing_rate_EI)
print(i_firing_rate_EI)
e_firing_rate_mean=np.array(e_firing_rate).mean()
e_firing_rate_EI_mean=np.array(e_firing_rate_EI).mean()
tmp=0
for i in range(N_E):
    tmp+=(e_firing_rate[i]-e_firing_rate_mean)*(e_firing_rate_EI[i]-e_firing_rate_EI_mean)
tmp1=0
tmp2=0
for i in range(N_E):
    tmp1+=(e_firing_rate[i]-e_firing_rate_mean)**2
    tmp2+=(e_firing_rate_EI[i]-e_firing_rate_EI_mean)**2
corr_coef_EI_E=tmp/(np.sqrt(tmp1)*np.sqrt(tmp2))
print(corr_coef_EI_E)

i_firing_rate_mean=np.array(i_firing_rate).mean()
i_firing_rate_EI_mean=np.array(i_firing_rate_EI).mean()
tmp=0
for i in range(N_I):
    tmp+=(i_firing_rate[i]-i_firing_rate_mean)*(i_firing_rate_EI[i]-i_firing_rate_EI_mean)
tmp1=0
tmp2=0
for i in range(N_I):
    tmp1+=(i_firing_rate[i]-i_firing_rate_mean)**2
    tmp2+=(i_firing_rate_EI[i]-i_firing_rate_EI_mean)**2
corr_coef_EI_I=tmp/(np.sqrt(tmp1)*np.sqrt(tmp2))
print(corr_coef_EI_I)

# Rewiring I-E
W_rIE=W.copy()
for i in range(N_E, n_neurons):
    for j in range(N_E):
        W[i, j] = np.random.lognormal(mu_IE, sigma_IE)
v = np.zeros((n_neurons, total_time))
h = np.zeros((n_neurons, total_time))
r = np.zeros(n_neurons)
e_firing_time_IE=[[] for i in range(N_E)]
i_firing_time_IE=[[] for i in range(N_I)]
e_firing_rate_IE=[]
i_firing_rate_IE=[]
for i in range(n_neurons):
    v[i, 0] = v_R
t = range(total_time - 1)
    # For excitatory neurons
for dt in t:

    for i in range(N_E):
        for j in range(n_neurons):
            h[i, dt] = h[i, dt] + W_rIE[i, j] * r[j]
        if v[i, dt] == spike:
            v[i, dt + 1] = v_R
        else:
            v[i, dt + 1] = v[i, dt] - v[i, dt] / tau_m + h[i, dt] + H_E / tau_m
            if v[i, dt + 1] >= theta:
                v[i, dt + 1] = spike
                r[i] = 1
                e_firing_time_IE[i].append(dt+1)
                #e_rates[IS-1,i]=e_rates[IS-1,i]+1
            else:
                r[i] = 0

    # firing_rate = e_spikes / total_time * 1000
    # e_firing_rates[IS-1,i]=firing_rate
# For inhibitory neurons
    for i in range(N_E, n_neurons):
        for j in range(n_neurons):
            h[i, dt] = h[i, dt] + W_rIE[i, j] * r[j]
        if v[i, dt] == spike:
            v[i, dt + 1] = v_R
        else:
            v[i, dt + 1] = v[i, dt] - v[i, dt] / tau_m + h[i, dt] + H_I / tau_m
            if v[i, dt + 1] >= theta:
                v[i, dt + 1] = spike
                r[i] = 1
                i_firing_time_IE[i-N_E].append(dt+1)
                # i_rates[IS-1,i-N_E]=i_rates[IS-1,i-N_E]+1
            else:
                r[i] = 0
for i in range(n_neurons):
    if i < N_E:
        if len(e_firing_time_IE[i])==1:
            e_firing_rate_IE.append(1)
        else:
            e_firing_rate_IE.append((e_firing_time_IE[i][-1]-e_firing_time_IE[i][0])/len(e_firing_time_IE[i]))
    else:
        if len(i_firing_time_IE[i-N_E])==1:
            i_firing_rate_IE.append(1)
        else:
            i_firing_rate_IE.append((i_firing_time_IE[i-N_E][-1]-i_firing_time_IE[i-N_E][0])/len(i_firing_time_IE[i-N_E]))
print(e_firing_rate_IE)
print(i_firing_rate_IE)
e_firing_rate_mean=np.array(e_firing_rate).mean()
e_firing_rate_IE_mean=np.array(e_firing_rate_IE).mean()
tmp=0
for i in range(N_E):
    tmp+=(e_firing_rate[i]-e_firing_rate_mean)*(e_firing_rate_IE[i]-e_firing_rate_IE_mean)
tmp1=0
tmp2=0
for i in range(N_E):
    tmp1+=(e_firing_rate[i]-e_firing_rate_mean)**2
    tmp2+=(e_firing_rate_IE[i]-e_firing_rate_IE_mean)**2
corr_coef_IE_E=tmp/(np.sqrt(tmp1)*np.sqrt(tmp2))
print(corr_coef_IE_E)

i_firing_rate_mean=np.array(i_firing_rate).mean()
i_firing_rate_IE_mean=np.array(i_firing_rate_IE).mean()
tmp=0
for i in range(N_I):
    tmp+=(i_firing_rate[i]-i_firing_rate_mean)*(i_firing_rate_IE[i]-i_firing_rate_IE_mean)
tmp1=0
tmp2=0
for i in range(N_I):
    tmp1+=(i_firing_rate[i]-i_firing_rate_mean)**2
    tmp2+=(i_firing_rate_IE[i]-i_firing_rate_IE_mean)**2
corr_coef_IE_I=tmp/(np.sqrt(tmp1)*np.sqrt(tmp2))
print(corr_coef_IE_I)
# Rewiring I-I
W_rII=W.copy()
for i in range(N_E, n_neurons):
    for j in range(N_E, n_neurons):
        W[i, j] = -np.random.lognormal(mu_II, sigma_II)
v = np.zeros((n_neurons, total_time))
h = np.zeros((n_neurons, total_time))
r = np.zeros(n_neurons)
e_firing_time_II=[[] for i in range(N_E)]
i_firing_time_II=[[] for i in range(N_I)]
e_firing_rate_II=[]
i_firing_rate_II=[]
for i in range(n_neurons):
    v[i, 0] = v_R
t = range(total_time - 1)
    # For excitatory neurons
for dt in t:

    for i in range(N_E):
        for j in range(n_neurons):
            h[i, dt] = h[i, dt] + W_rII[i, j] * r[j]
        if v[i, dt] == spike:
            v[i, dt + 1] = v_R
        else:
            v[i, dt + 1] = v[i, dt] - v[i, dt] / tau_m + h[i, dt] + H_E / tau_m
            if v[i, dt + 1] >= theta:
                v[i, dt + 1] = spike
                r[i] = 1
                e_firing_time_II[i].append(dt+1)
                #e_rates[IS-1,i]=e_rates[IS-1,i]+1
            else:
                r[i] = 0

    # firing_rate = e_spikes / total_time * 1000
    # e_firing_rates[IS-1,i]=firing_rate
# For inhibitory neurons
    for i in range(N_E, n_neurons):
        for j in range(n_neurons):
            h[i, dt] = h[i, dt] + W_rII[i, j] * r[j]
        if v[i, dt] == spike:
            v[i, dt + 1] = v_R
        else:
            v[i, dt + 1] = v[i, dt] - v[i, dt] / tau_m + h[i, dt] + H_I / tau_m
            if v[i, dt + 1] >= theta:
                v[i, dt + 1] = spike
                r[i] = 1
                i_firing_time_II[i-N_E].append(dt+1)
                # i_rates[IS-1,i-N_E]=i_rates[IS-1,i-N_E]+1
            else:
                r[i] = 0
for i in range(n_neurons):
    if i < N_E:
        if len(e_firing_time_IE[i])==1:
            e_firing_rate_II.append(1)
        else:
            e_firing_rate_II.append((e_firing_time_II[i][-1]-e_firing_time_II[i][0])/len(e_firing_time_II[i]))
    else:
        if len(i_firing_time_II[i-N_E])==1:
            i_firing_rate_II.append(1)
        else:
            i_firing_rate_II.append((i_firing_time_II[i-N_E][-1]-i_firing_time_II[i-N_E][0])/len(i_firing_time_II[i-N_E]))
print(e_firing_rate_II)
print(i_firing_rate_II)
e_firing_rate_mean=np.array(e_firing_rate).mean()
e_firing_rate_II_mean=np.array(e_firing_rate_II).mean()
tmp=0
for i in range(N_E):
    tmp+=(e_firing_rate[i]-e_firing_rate_mean)*(e_firing_rate_II[i]-e_firing_rate_II_mean)
tmp1=0
tmp2=0
for i in range(N_E):
    tmp1+=(e_firing_rate[i]-e_firing_rate_mean)**2
    tmp2+=(e_firing_rate_II[i]-e_firing_rate_II_mean)**2
corr_coef_II_E=tmp/(np.sqrt(tmp1)*np.sqrt(tmp2))
print(corr_coef_II_E)

i_firing_rate_mean=np.array(i_firing_rate).mean()
i_firing_rate_II_mean=np.array(i_firing_rate_II).mean()
tmp=0
for i in range(N_I):
    tmp+=(i_firing_rate[i]-i_firing_rate_mean)*(i_firing_rate_II[i]-i_firing_rate_II_mean)
tmp1=0
tmp2=0
for i in range(N_I):
    tmp1+=(i_firing_rate[i]-i_firing_rate_mean)**2
    tmp2+=(i_firing_rate_II[i]-i_firing_rate_II_mean)**2
corr_coef_II_I=tmp/(np.sqrt(tmp1)*np.sqrt(tmp2))
print(corr_coef_II_I)

# Theoretical method: Equation (32)
W_EE_mean=np.mat(W_rEE[0:N_E,0:N_E]).mean()
W_EI_mean=np.mat(W_rEI[0:N_E,N_E:n_neurons]).mean()
W_IE_mean=np.mat(W_rIE[N_E:n_neurons,0:N_E]).mean()
W_II_mean=np.mat(W_rII[N_E:n_neurons,N_E:n_neurons]).mean()
W_rEE2=np.mat([[W_rEE[i][j]**2 for j in range(n_neurons)] for i in range(n_neurons)])
W_EE2_mean=W_rEE2[0:N_E, 0:N_E].mean()
W_rEI2=np.mat([[W_rEI[i][j]**2 for j in range(n_neurons)] for i in range(n_neurons)])
W_EI2_mean=W_rEI2[0:N_E, N_E:n_neurons].mean()
W_rIE2=np.mat([[W_rIE[i][j]**2 for j in range(n_neurons)] for i in range(n_neurons)])
W_IE2_mean=W_rIE2[N_E:n_neurons, 0:N_E].mean()
W_rII2=np.mat([[W_rII[i][j]**2 for j in range(n_neurons)] for i in range(n_neurons)])
W_II2_mean=W_rII2[N_E:n_neurons, N_E:n_neurons].mean()
v_original_E_mean=np.array(e_firing_rate).mean()
v_original_I_mean=np.array(i_firing_rate).mean()
e_firing_rate2=[e_firing_rate[i]**2 for i in range(N_E)]
i_firing_rate2=[i_firing_rate[i]**2 for i in range(N_I)]
v_original_E2_mean=np.array(e_firing_rate2).mean()
v_original_I2_mean=np.array(i_firing_rate2).mean()
sE2=W_EE2*v_original_E2_mean-(W_EE**2)*(v_original_E_mean**2)+W_EI2*v_original_I2_mean-(W_EI**2)*(v_original_I_mean**2)
sI2=W_IE2*v_original_E2_mean-(W_IE**2)*(v_original_E_mean**2)+W_II2*v_original_I2_mean-(W_II**2)*(v_original_I_mean**2)
# 1. E-E Rewiring

v_rEE_E_mean=e_firing_rate_EE_mean
e_firing_rate_EE2=[e_firing_rate_EE[i]**2 for i in range(N_E)]
v_rEE_E2_mean=np.array(e_firing_rate_EE2).mean()
v_rEE_I_mean=i_firing_rate_EE_mean
i_firing_rate_EE2=[i_firing_rate_EE[i]**2 for i in range(N_I)]
v_rEE_I2_mean=np.array(i_firing_rate_EE2).mean()
W_rEE_tilde=np.multiply(W[0:N_E,0:N_E],W_rEE[0:N_E,0:N_E]).mean()
v_rEE_E_tilde=np.multiply(e_firing_rate,e_firing_rate_EE).mean()
v_rEE_I_tilde=np.multiply(i_firing_rate,i_firing_rate_EE).mean()
rho_E_rEE=(-(W_EE**2)*(v_rEE_E_mean**2)+W_rEE_tilde*v_rEE_E_tilde-(W_EI**2)*(v_rEE_I_mean**2)+W_EI2*v_rEE_I_tilde)/sE2
rho_I_rEE=(-(W_IE**2)*(v_rEE_E_mean**2)+W_IE2*v_rEE_E_tilde-(W_II**2)*(v_rEE_I_mean**2)+W_II2*v_rEE_E_tilde)/sI2

# 2. E-I Rewiring

v_rEI_E_mean=e_firing_rate_EI_mean
e_firing_rate_EI2=[e_firing_rate_EI[i]**2 for i in range(N_E)]
v_rEI_E2_mean=np.array(e_firing_rate_EI2).mean()
v_rEI_I_mean=i_firing_rate_EI_mean
i_firing_rate_EI2=[i_firing_rate_EI[i]**2 for i in range(N_I)]
v_rEI_I2_mean=np.array(i_firing_rate_EI2).mean()
W_rEI_tilde=np.multiply(W[0:N_E,N_E:n_neurons],W_rEI[0:N_E,N_E:n_neurons]).mean()
v_rEI_E_tilde=np.multiply(e_firing_rate,e_firing_rate_EI).mean()
v_rEI_I_tilde=np.multiply(i_firing_rate,i_firing_rate_EI).mean()
rho_E_rEI=(-(W_EE**2)*(v_rEI_E_mean**2)+W_EE2*v_rEI_E_tilde-(W_EI**2)*(v_rEI_I_mean**2)+W_rEI_tilde*v_rEI_I_tilde)/sE2
rho_I_rEI=(-(W_IE**2)*(v_rEI_E_mean**2)+W_IE2*v_rEI_E_tilde-(W_II**2)*(v_rEI_I_mean**2)+W_II2*v_rEI_I_tilde)/sI2

# 3. I-E Rewiring

v_rIE_E_mean=e_firing_rate_IE_mean
e_firing_rate_IE2=[e_firing_rate_IE[i]**2 for i in range(N_E)]
v_rIE_E2_mean=np.array(e_firing_rate_IE2).mean()
v_rIE_I_mean=i_firing_rate_IE_mean
i_firing_rate_IE2=[i_firing_rate_IE[i]**2 for i in range(N_I)]
v_rIE_I2_mean=np.array(i_firing_rate_IE2).mean()
W_rIE_tilde=np.multiply(W[N_E:n_neurons,0:N_E],W_rIE[N_E:n_neurons,0:N_E]).mean()
v_rIE_E_tilde=np.multiply(e_firing_rate,e_firing_rate_IE).mean()
v_rIE_I_tilde=np.multiply(i_firing_rate,i_firing_rate_IE).mean()
rho_E_rIE=(-(W_EE**2)*(v_rIE_E_mean**2)+W_EE2*v_rIE_E_tilde-(W_EI**2)*(v_rIE_I_mean**2)+W_EI2*v_rIE_I_tilde)/sE2
rho_I_rIE=(-(W_IE**2)*(v_rIE_E_mean**2)+W_rIE_tilde*v_rIE_E_tilde-(W_II**2)*(v_rIE_I_mean**2)+W_II2*v_rIE_I_tilde)/sI2

# 4. I-I Rewiring
v_rII_E_mean=e_firing_rate_II_mean
e_firing_rate_II2=[e_firing_rate_II[i]**2 for i in range(N_E)]
v_rII_E2_mean=np.array(e_firing_rate_II2).mean()
v_rII_I_mean=i_firing_rate_II_mean
i_firing_rate_II2=[i_firing_rate_II[i]**2 for i in range(N_I)]
v_rII_I2_mean=np.array(i_firing_rate_II2).mean()
W_rII_tilde=np.multiply(W[N_E:n_neurons,N_E:n_neurons],W_rII[N_E:n_neurons,N_E:n_neurons]).mean()
v_rII_E_tilde=np.multiply(e_firing_rate,e_firing_rate_II).mean()
v_rII_I_tilde=np.multiply(i_firing_rate,i_firing_rate_II).mean()
rho_E_rII=(-(W_EE**2)*(v_rII_E_mean**2)+W_EE2*v_rII_E_tilde-(W_EI**2)*(v_rII_I_mean**2)+W_EI2*v_rII_I_tilde)/sE2
rho_I_rII=(-(W_IE**2)*(v_rII_E_mean**2)+W_IE2*v_rII_E_tilde-(W_II**2)*(v_rII_I_mean**2)+W_rII_tilde*v_rII_I_tilde)/sI2
#Plotting figure 3
plt.figure()

plt.subplot(1,4,1)
plt.ylim([-1,1])   
plt.scatter([1,2],[corr_coef_EE_E,corr_coef_EE_I])
plt.bar([1,2],[rho_E_rEE,rho_I_rEE])
plt.subplot(1,4,2)
plt.ylim([-1,1])   
plt.scatter([1,2],[corr_coef_EI_E,corr_coef_EI_I])
plt.bar([1,2],[rho_E_rEI,rho_I_rIE])
plt.subplot(1,4,3)
plt.ylim([-1,1])   
plt.scatter([1,2],[corr_coef_IE_E,corr_coef_IE_I])
plt.bar([1,2],[rho_E_rIE,rho_I_rIE])
plt.subplot(1,4,4)
plt.ylim([-1,1])   
plt.scatter([1,2],[corr_coef_II_E,corr_coef_II_I])
plt.bar([1,2],[rho_E_rII,rho_I_rII])

# u_E=np.sqrt(N_E)*(H_E+W_EE*v_original_E_mean-np.sqrt(N_I)/np.sqrt(N_E)*W_EI*v_original_I_mean)
# u_I=np.sqrt(N_E)*(H_I+W_IE*v_original_E_mean-np.sqrt(N_I)/np.sqrt(N_E)*W_II*v_original_I_mean)

s_EE2=W_EE2*v_original_E2_mean-(W_EE**2)*(v_original_E_mean**2)
s_rEE2=W_EE2_mean*v_rEE_E2_mean-(W_EE_mean**2)*(v_rEE_E_mean**2)
s_EI2=W_EI2*v_original_I2_mean-(W_EI**2)*(v_original_I_mean**2)
s_rEI2=W_EI2_mean*v_rEI_I2_mean-(W_EI_mean**2)*(v_rEI_I_mean**2)
s_IE2=W_IE2*v_original_E2_mean-(W_IE**2)*(v_original_E_mean**2)
s_rIE2=W_IE2_mean*v_rIE_E2_mean-(W_IE_mean**2)*(v_rIE_E_mean**2)
s_II2=W_II2*v_original_I2_mean-(W_II**2)*(v_original_I_mean**2)
s_rII2=W_II2_mean*v_rII_I2_mean-(W_II_mean**2)*(v_rII_I_mean**2)
u_rEE_E=np.sqrt(N_E)*(H_E+W_EE_mean*v_rEE_E_mean-np.sqrt(N_I)/np.sqrt(N_E)*W_EI*v_rEE_I_mean)
u_rEI_E=np.sqrt(N_E)*(H_E+W_EE*v_rEI_E_mean-np.sqrt(N_I)/np.sqrt(N_E)*W_EI_mean*v_rEI_I_mean)
u_rIE_I=np.sqrt(N_E)*(H_I+W_IE_mean*v_rIE_E_mean-np.sqrt(N_I)/np.sqrt(N_E)*W_II*v_rIE_I_mean)
u_rII_I=np.sqrt(N_E)*(H_I+W_IE*v_rII_E_mean-np.sqrt(N_I)/np.sqrt(N_E)*W_II_mean*v_rII_I_mean)
n_trials=500
h_EE=np.zeros(n_trials)
h_EI=np.zeros(n_trials)
h_IE=np.zeros(n_trials)
h_II=np.zeros(n_trials)
for i in range(n_trials):
    eta_E=np.random.normal(0,1)
    eta_I=np.random.normal(0,1)
    h_EE[i]=u_rEE_E+eta_E*np.sqrt(s_rEE2)+eta_I*np.sqrt(s_EI2)
    h_EI[i]=u_rEI_E+eta_E*np.sqrt(s_EE2)+eta_I*np.sqrt(s_rEI2)
    h_IE[i]=u_rIE_I+eta_E*np.sqrt(s_rIE2)+eta_I*np.sqrt(s_II2)
    h_II[i]=u_rII_I+eta_E*np.sqrt(s_IE2)+eta_I*np.sqrt(s_rII2)
plt.figure()
# plt.xlim(0,2000)
sns.distplot(h_EE,bins=50,kde=True,color='b')
sns.distplot(h_EI,bins=50,kde=True,color='r')
# kde_EE = KernelDensity(kernel='gaussian', bandwidth=0.005).fit(h_EE.reshape(-1,1))
# kde_EI = KernelDensity(kernel='gaussian', bandwidth=0.05).fit(h_EI.reshape(-1,1))
# plt.plot(x,np.exp(kde_EE.score_samples(x)))
# plt.plot(x,np.exp(kde_EI.score_samples(x)))
plt.figure()
# plt.xlim(0,2000)
sns.distplot(h_IE,bins=50,kde=True,color='b')
sns.distplot(h_II,bins=50,kde=True,color='r')
# kde_IE = KernelDensity(kernel='gaussian', bandwidth=0.05).fit(h_IE.reshape(-1,1))
# kde_II = KernelDensity(kernel='gaussian', bandwidth=0.05).fit(h_II.reshape(-1,1))
# plt.plot(x,np.exp(kde_IE.score_samples(x)))
# plt.plot(x,np.exp(kde_II.score_samples(x)))