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
W_EE_mean=np.mat(W[0:N_E,0:N_E]).mean()
W2=[[W[i][j]**2 for j in range(n_neurons)] for i in range(n_neurons)]
W_EE2_mean=np.mat(W2[0:N_E,0:N_E]).mean()
W_EI_mean=np.mat(W[0:N_E,N_E:n_neurons]).mean()
W_EI2_mean=np.mat(W2[0:N_E,N_E:n_neurons]).mean()
W_IE_mean=np.mat(W[N_E:n_neurons,0:N_E]).mean()
W_IE2_mean=np.mat(W2[N_E:n_neurons,0:N_E]).mean()
W_II_mean=np.mat(W[N_E:n_neurons,N_E:n_neurons]).mean()
W_II2_mean=np.mat(W2[N_E:n_neurons,N_E:n_neurons]).mean()