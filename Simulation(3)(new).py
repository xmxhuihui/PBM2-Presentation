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
from scipy.stats import pearsonr
# Number of excitatory and inhibitory neurons
N_E = 400
N_I = 200
n_neurons = N_E + N_I
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
lnmu_EI=np.log(mu_EI**2/np.sqrt(mu_EI**2+sigma_EI2))
lnmu_IE=np.log(mu_IE**2/np.sqrt(mu_IE**2+sigma_IE2))
lnmu_II=np.log(mu_II**2/np.sqrt(mu_II**2+sigma_II2))
lnsigma_EI=np.sqrt(np.log(1+sigma_EI2/mu_EI**2))
lnsigma_IE=np.sqrt(np.log(1+sigma_IE2/mu_IE**2))
lnsigma_II=np.sqrt(np.log(1+sigma_II2/mu_II**2))
theta = 33
tau_m = 10
H_E = 35
H_I = 33.8
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
c_EE = 0.2
c_EI = 0.4
c_IE = 0.3
c_II = 0.4
W = np.zeros((n_neurons, n_neurons))

for i in range(n_neurons):
    for j in range(n_neurons):
        if i < N_E:
            # E -> E
            if j < N_E:
                if random.uniform(0, 1) <= c_EE:
                    index = random.randint(1, 1420)
                    W[i, j] = spines_info['Volume'].loc[spines_info['Global_SpineID'] == index].values[0] * g
            # E -> I
            else:
                if random.uniform(0, 1) <= c_EI:
                    W[i, j] = -np.random.lognormal(lnmu_EI, lnsigma_EI)
        else:
            # I -> E
            if j < N_E:
                if random.uniform(0, 1) <= c_IE:
                    W[i, j] = np.random.lognormal(lnmu_IE, lnsigma_IE)
            # I -> I
            else:
                if random.uniform(0, 1) <= c_II:
                    W[i, j] = -np.random.lognormal(lnmu_II, lnsigma_II)

v_original = np.zeros((n_neurons, total_time+1))
h = np.zeros((n_neurons, total_time+1))
r = np.zeros(n_neurons)
e_firing_time=[[] for i in range(N_E)]
i_firing_time=[[] for i in range(N_I)]
e_firing_rate=[]
i_firing_rate=[]
H=np.zeros(n_neurons)
for i in range(n_neurons):
    if(i<N_E):
        H[i]=H_E
    else:
        H[i]=H_I
# Recording the state of each neuron in the last timestep
for i in range(n_neurons):
    v_original[i, 0] = v_R
t = range(total_time)
    # For excitatory neurons
for dt in t:
    h[:, dt] =np.dot(W,np.transpose(r))
    v_original[:,dt+1]=v_original[:,dt]+(-v_original[:,dt]/tau_m+h[:,dt]+H/tau_m)*0.1
    for i in range(n_neurons):
        if v_original[i,dt]==spike:
            v_original[i,dt+1]=v_R
        if v_original[i,dt+1]>=theta:
            v_original[i,dt+1]=spike
            r[i]=1
            if i < N_E:
                e_firing_time[i].append(dt + 1)
            else:
                i_firing_time[i - N_E].append(dt + 1)
        else:
            r[i]=0
for j in range(n_neurons):
    if j<N_E and len(e_firing_time[j]) != 0:
        e_firing_rate.append(len(e_firing_time[j]) / total_time * 1000*10)
    elif j>=N_E and len(i_firing_time[j-N_E]) != 0:
        i_firing_rate.append(len(i_firing_time[j-N_E]) / total_time * 1000*10)
# Rewiring E-E
W_rEE=W.copy()
for i in range(N_E):
    for j in range(N_E):
        W_rEE[i,j]=0
        if random.uniform(0, 1) <= c_EE:
            index = random.randint(1, 1420)
            W_rEE[i, j] = spines_info['Volume'].loc[spines_info['Global_SpineID'] == index].values[0] * g
v = np.zeros((n_neurons, total_time+1))
h = np.zeros((n_neurons, total_time+1))
r = np.zeros(n_neurons)
e_firing_time_EE=[[] for i in range(N_E)]
i_firing_time_EE=[[] for i in range(N_I)]
e_firing_rate_EE=[]
i_firing_rate_EE=[]
for i in range(n_neurons):
        v[i, 0] = v_R
t = range(total_time)

    # For excitatory neurons
for dt in t:
    h[:, dt] =np.dot(W_rEE,np.transpose(r))
    v[:,dt+1]=v[:,dt]+(-v[:,dt]/tau_m+h[:,dt]+H/tau_m)*0.1
    for i in range(n_neurons):
        if v[i,dt]==spike:
            v[i,dt+1]=v_R
        if v[i,dt+1]>=theta:
            v[i,dt+1]=spike
            r[i]=1
            if i < N_E:
                e_firing_time_EE[i].append(dt + 1)
            else:
                i_firing_time_EE[i - N_E].append(dt + 1)
        else:
            r[i]=0
for j in range(n_neurons):
    if j<N_E and len(e_firing_time_EE[j]) != 0:
        e_firing_rate_EE.append(len(e_firing_time_EE[j]) / total_time * 1000*10)
    elif j>=N_E and len(i_firing_time_EE[j-N_E]) != 0:
        i_firing_rate_EE.append(len(i_firing_time_EE[j-N_E]) / total_time * 1000*10)
e_firing_rate_mean=np.array(e_firing_rate).mean()
e_firing_rate_EE_mean=np.array(e_firing_rate_EE).mean()

i_firing_rate_mean=np.array(i_firing_rate).mean()
i_firing_rate_EE_mean=np.array(i_firing_rate_EE).mean()

corr_coef_EE_E=pearsonr(e_firing_rate,e_firing_rate_EE)
corr_coef_EE_I=pearsonr(i_firing_rate,i_firing_rate_EE)
# Rewiring E_I
W_rEI=W.copy()
for i in range(N_E):
    for j in range(N_E, n_neurons):
        W_rEI[i, j] = -np.random.lognormal(lnmu_EI, lnsigma_EI)
v = np.zeros((n_neurons, total_time+1))
h = np.zeros((n_neurons, total_time+1))
r = np.zeros(n_neurons)
e_firing_time_EI=[[] for i in range(N_E)]
i_firing_time_EI=[[] for i in range(N_I)]
e_firing_rate_EI=[]
i_firing_rate_EI=[]
for i in range(n_neurons):
    v[i, 0] = v_R
t = range(total_time)
    # For excitatory neurons
for dt in t:
    h[:, dt] =np.dot(W_rEI,np.transpose(r))
    v[:,dt+1]=v[:,dt]+(-v[:,dt]/tau_m+h[:,dt]+H/tau_m)*0.1
    for i in range(n_neurons):
        if v[i,dt]==spike:
            v[i,dt+1]=v_R
        if v[i,dt+1]>=theta:
            v[i,dt+1]=spike
            r[i]=1
            if i < N_E:
                e_firing_time_EI[i].append(dt + 1)
            else:
                i_firing_time_EI[i - N_E].append(dt + 1)
        else:
            r[i]=0
for j in range(n_neurons):
    if j<N_E and len(e_firing_time_EI[j]) != 0:
        e_firing_rate_EI.append(len(e_firing_time_EI[j]) / total_time * 1000*10)
    elif j>=N_E and len(i_firing_time_EI[j-N_E]) != 0:
        i_firing_rate_EI.append(len(i_firing_time_EI[j-N_E]) / total_time * 1000*10)
e_firing_rate_mean=np.array(e_firing_rate).mean()
e_firing_rate_EI_mean=np.array(e_firing_rate_EI).mean()

i_firing_rate_mean=np.array(i_firing_rate).mean()
i_firing_rate_EI_mean=np.array(i_firing_rate_EI).mean()

corr_coef_EI_E=pearsonr(e_firing_rate,e_firing_rate_EI)
corr_coef_EI_I=pearsonr(i_firing_rate,i_firing_rate_EI)
# Rewiring I-E
W_rIE=W.copy()
for i in range(N_E, n_neurons):
    for j in range(N_E):
        W_rIE[i, j] = np.random.lognormal(lnmu_IE, lnsigma_IE)
v = np.zeros((n_neurons, total_time+1))
h = np.zeros((n_neurons, total_time+1))
r = np.zeros(n_neurons)
e_firing_time_IE=[[] for i in range(N_E)]
i_firing_time_IE=[[] for i in range(N_I)]
e_firing_rate_IE=[]
i_firing_rate_IE=[]
for i in range(n_neurons):
    v[i, 0] = v_R
t = range(total_time)
    # For excitatory neurons
for dt in t:
    h[:, dt] =np.dot(W_rIE,np.transpose(r))
    v[:,dt+1]=v[:,dt]+(-v[:,dt]/tau_m+h[:,dt]+H/tau_m)*0.1
    for i in range(n_neurons):
        if v[i,dt]==spike:
            v[i,dt+1]=v_R
        if v[i,dt+1]>=theta:
            v[i,dt+1]=spike
            r[i]=1
            if i < N_E:
                e_firing_time_IE[i].append(dt + 1)
            else:
                i_firing_time_IE[i - N_E].append(dt + 1)
        else:
            r[i]=0
for j in range(n_neurons):
    if j<N_E and len(e_firing_time_IE[j]) != 0:
        e_firing_rate_IE.append(len(e_firing_time_IE[j]) / total_time * 1000*10)
    elif j>=N_E and len(i_firing_time_IE[j-N_E]) != 0:
        i_firing_rate_IE.append(len(i_firing_time_IE[j-N_E]) / total_time * 1000*10)
e_firing_rate_mean=np.array(e_firing_rate).mean()
e_firing_rate_IE_mean=np.array(e_firing_rate_IE).mean()

i_firing_rate_mean=np.array(i_firing_rate).mean()
i_firing_rate_IE_mean=np.array(i_firing_rate_IE).mean()
corr_coef_IE_E=pearsonr(e_firing_rate,e_firing_rate_IE)
corr_coef_IE_I=pearsonr(i_firing_rate,i_firing_rate_IE)
# Rewiring I-I
W_rII=W.copy()
for i in range(N_E, n_neurons):
    for j in range(N_E, n_neurons):
        W_rII[i, j] = -np.random.lognormal(lnmu_II, lnsigma_II)
v = np.zeros((n_neurons, total_time+1))
h = np.zeros((n_neurons, total_time+1))
r = np.zeros(n_neurons)
e_firing_time_II=[[] for i in range(N_E)]
i_firing_time_II=[[] for i in range(N_I)]
e_firing_rate_II=[]
i_firing_rate_II=[]
for i in range(n_neurons):
    v[i, 0] = v_R
t = range(total_time)
    # For excitatory neurons
for dt in t:
    h[:, dt] =np.dot(W_rII,np.transpose(r))
    v[:,dt+1]=v[:,dt]+(-v[:,dt]/tau_m+h[:,dt]+H/tau_m)*0.1
    for i in range(n_neurons):
        if v[i,dt]==spike:
            v[i,dt+1]=v_R
        if v[i,dt+1]>=theta:
            v[i,dt+1]=spike
            r[i]=1
            if i < N_E:
                e_firing_time_II[i].append(dt + 1)
            else:
                i_firing_time_II[i - N_E].append(dt + 1)
        else:
            r[i]=0
for j in range(n_neurons):
    if j<N_E and len(e_firing_time_II[j]) != 0:
        e_firing_rate_II.append(len(e_firing_time_II[j]) / total_time * 1000*10)
    elif j>=N_E and len(i_firing_time_II[j-N_E]) != 0:
        i_firing_rate_II.append(len(i_firing_time_II[j-N_E]) / total_time * 1000*10)
# print(e_firing_rate_II)
# print(i_firing_rate_II)
e_firing_rate_mean=np.array(e_firing_rate).mean()
e_firing_rate_II_mean=np.array(e_firing_rate_II).mean()

i_firing_rate_mean=np.array(i_firing_rate).mean()
i_firing_rate_II_mean=np.array(i_firing_rate_II).mean()
corr_coef_II_E=pearsonr(e_firing_rate,e_firing_rate_II)
corr_coef_II_I=pearsonr(i_firing_rate,i_firing_rate_II)
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
v_original_E_mean=e_firing_rate_mean
v_original_I_mean=i_firing_rate_mean
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
rho_I_rEE=(-(W_IE**2)*(v_rEE_E_mean**2)+W_IE2*v_rEE_E_tilde-(W_II**2)*(v_rEE_I_mean**2)+W_II2*v_rEE_I_tilde)/sI2

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
plt.bar([1,2],[rho_E_rEE,rho_I_rEE])

plt.subplot(1,4,3)
plt.ylim([-1,1])   
plt.bar([1,2],[rho_E_rEI,rho_I_rEI])

plt.subplot(1,4,2)
plt.ylim([-1,1])   
plt.bar([1,2],[rho_E_rIE,rho_I_rIE])

plt.subplot(1,4,4)
plt.ylim([-1,1])   
plt.bar([1,2],[rho_E_rII,rho_I_rII])


plt.figure()
plt.subplot(1,4,1)
plt.ylim([-1,1])  
plt.scatter([1,2],[corr_coef_EE_E[0],corr_coef_EE_I[0]])
plt.subplot(1,4,3)
plt.ylim([-1,1]) 
plt.scatter([1,2],[corr_coef_EI_E[0],corr_coef_EI_I[0]])
plt.subplot(1,4,2)
plt.ylim([-1,1])
plt.scatter([1,2],[corr_coef_IE_E[0],corr_coef_IE_I[0]])
plt.subplot(1,4,4)
plt.ylim([-1,1])   
plt.scatter([1,2],[corr_coef_II_E[0],corr_coef_II_I[0]])
# u_E=np.sqrt(N_E)*(H_E+W_EE*v_original_E_mean-np.sqrt(N_I)/np.sqrt(N_E)*W_EI*v_original_I_mean)
# u_I=np.sqrt(N_E)*(H_I+W_IE*v_original_E_mean-np.sqrt(N_I)/np.sqrt(N_E)*W_II*v_original_I_mean)

