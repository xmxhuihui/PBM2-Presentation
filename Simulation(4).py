#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 10 12:22:47 2020

@author: xmxhuihui
"""
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns

N_E = 20
N_I = 10
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

n_trials=2000
h_E=np.zeros(n_trials)
h_I=np.zeros(n_trials)
for k in range(n_trials):
    W = np.zeros((n_neurons, n_neurons))
    for i in range(n_neurons):
        for j in range(n_neurons):
            if i < N_E:
                # E -> E
                if j < N_E:
                    if random.uniform(0, 1) <= c_EE:
                        index = random.randint(1, 1420)
                        W[i, j] = spines_info['Volume'].loc[spines_info['Global_SpineID'] == index].values[0] * g
                # I -> E
                else:
                    W[i, j] = -np.random.lognormal(lnmu_EI, lnsigma_EI)
            else:
                # E -> I
                if j < N_E:
                    if random.uniform(0, 1) <= c_IE:
                        W[i, j] = np.random.lognormal(lnmu_IE, lnsigma_IE)
                # I -> I
                else:
                    if random.uniform(0, 1) <= c_II:
                        W[i, j] = -np.random.lognormal(lnmu_II, lnsigma_II)
    v = np.zeros((n_neurons, total_time+1))
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
        v[i, 0] = v_R
    t = range(total_time)
        # For excitatory neurons
    for dt in t:
        h[:, dt] =np.dot(W,np.transpose(r))
        v[:,dt+1]=v[:,dt]+(-v[:,dt]/tau_m+h[:,dt]+H/tau_m)*0.1
        for i in range(n_neurons):
            if v[i,dt]==spike:
                v[i,dt+1]=v_R
            if v[i,dt+1]>=theta:
                v[i,dt+1]=spike
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
            
    W_EE_mean=np.mat(W[0:N_E,0:N_E]).mean()
    W_EI_mean=np.mat(W[0:N_E,N_E:n_neurons]).mean()
    W_IE_mean=np.mat(W[N_E:n_neurons,0:N_E]).mean()
    W_II_mean=np.mat(W[N_E:n_neurons,N_E:n_neurons]).mean()
    W2=np.mat([[W[i][j]**2 for j in range(n_neurons)] for i in range(n_neurons)])
    W_EE2_mean=W2[0:N_E, 0:N_E].mean()
    W_EI2_mean=W2[0:N_E, N_E:n_neurons].mean()
    W_IE2_mean=W[N_E:n_neurons, 0:N_E].mean()
    W_II2_mean=W[N_E:n_neurons, N_E:n_neurons].mean()
    e_firing_rate2=[e_firing_rate[i]**2 for i in range(N_E)]
    i_firing_rate2=[i_firing_rate[i]**2 for i in range(N_I)]
    v_E_mean=np.array(e_firing_rate).mean()
    v_E2_mean=np.array(e_firing_rate2).mean()
    v_I_mean=np.array(i_firing_rate).mean()
    v_I2_mean=np.array(i_firing_rate2).mean()
    s_EE2=W_EE2*v_E2_mean-(W_EE**2)*(v_E_mean**2)
    s_EI2=W_EI2*v_I2_mean-(W_EI**2)*(v_I_mean**2)
    s_IE2=W_IE2*v_E2_mean-(W_IE**2)*(v_E_mean**2)
    s_II2=W_II2*v_I2_mean-(W_II**2)*(v_I_mean**2)
    u_E=np.sqrt(N_E)*(H_E+W_EE_mean*v_E_mean-np.sqrt(N_I)/np.sqrt(N_E)*W_EI_mean*v_I_mean)
    u_I=np.sqrt(N_E)*(H_E+W_IE_mean*v_E_mean-np.sqrt(N_I)/np.sqrt(N_E)*W_II_mean*v_I_mean)
# s_EE2=W_EE2*v_original_E2_mean-(W_EE**2)*(v_original_E_mean**2)
# s_rEE2=W_EE2_mean*v_rEE_E2_mean-(W_EE_mean**2)*(v_rEE_E_mean**2)
# s_EI2=W_EI2*v_original_I2_mean-(W_EI**2)*(v_original_I_mean**2)
# s_rEI2=W_EI2_mean*v_rEI_I2_mean-(W_EI_mean**2)*(v_rEI_I_mean**2)
# s_IE2=W_IE2*v_original_E2_mean-(W_IE**2)*(v_original_E_mean**2)
# s_rIE2=W_IE2_mean*v_rIE_E2_mean-(W_IE_mean**2)*(v_rIE_E_mean**2)
# s_II2=W_II2*v_original_I2_mean-(W_II**2)*(v_original_I_mean**2)
# s_rII2=W_II2_mean*v_rII_I2_mean-(W_II_mean**2)*(v_rII_I_mean**2)
# u_rEE_E=np.sqrt(N_E)*(H_E+W_EE_mean*v_rEE_E_mean-np.sqrt(N_I)/np.sqrt(N_E)*W_EI*v_rEE_I_mean)
# u_rEI_E=np.sqrt(N_E)*(H_E+W_EE*v_rEI_E_mean-np.sqrt(N_I)/np.sqrt(N_E)*W_EI_mean*v_rEI_I_mean)
# u_rIE_I=np.sqrt(N_E)*(H_I+W_IE_mean*v_rIE_E_mean-np.sqrt(N_I)/np.sqrt(N_E)*W_II*v_rIE_I_mean)
# u_rII_I=np.sqrt(N_E)*(H_I+W_IE*v_rII_E_mean-np.sqrt(N_I)/np.sqrt(N_E)*W_II_mean*v_rII_I_mean)

# h_IE=np.zeros(n_trials)
# h_II=np.zeros(n_trials)

    eta_E=np.random.normal(0,1)
    eta_I=np.random.normal(0,1)
    h_E[k]=(u_E+eta_E*np.sqrt(s_EE2)+eta_I*np.sqrt(s_EI2))
    h_I[k]=(u_I+eta_E*np.sqrt(s_IE2)+eta_I*np.sqrt(s_II2))
# h_IE[i]=(u_rIE_I+eta_E*np.sqrt(s_rIE2)+eta_I*np.sqrt(s_II2))/N_I
# h_II[i]=(u_rII_I+eta_E*np.sqrt(s_IE2)+eta_I*np.sqrt(s_rII2))/N_I

plt.figure()
# plt.xlim(0,2000)
sns.distplot(h_E,bins=50,kde=True,color='b')
sns.distplot(h_I,bins=50,kde=True,color='r')
# plt.figure()
# # plt.xlim(0,2000)
# sns.distplot(h_IE,bins=50,kde=True,color='b')
# sns.distplot(h_II,bins=50,kde=True,color='r')