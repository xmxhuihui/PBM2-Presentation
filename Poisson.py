#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 19:43:12 2020

@author: xmxhuihui
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random


N_E=32000
N_I=8000
n_neurons = N_E + N_I
total_time=3
e_intervals=[[]for i in range(N_E)]
i_intervals=[[]for i in range(N_I)]
e_rate=0.5
i_rate=5
e_firing_rates=[]
i_firing_rates=[]
for i in range(N_E):
    sum=0
    while(sum<=total_time):
        e_interval=np.random.exponential(scale=1/e_rate)
        e_intervals[i].append(e_interval)
        sum=np.sum(e_intervals[i])
    e_firing_rates.append(len(e_intervals[i])/sum)
for i in range(N_I):
    sum=0
    while(sum<=total_time):
        i_interval=np.random.exponential(scale=1/i_rate)
        i_intervals[i].append(i_interval)
        sum=np.sum(i_intervals[i])
    i_firing_rates.append(len(i_intervals[i])/sum)
plt.figure()
plt.xlim(0,5)
plt.xlabel('Firing rate(Hz)')
plt.ylabel('Probability density($Hz^{-1}$)')
sns.distplot(e_firing_rates,color='b')
plt.figure()
plt.xlim(0,30)
plt.xlabel('Firing rate(Hz)')
plt.ylabel('Probability density($Hz^{-1}$)')
sns.distplot(i_firing_rates,color='r')
plt.figure()
#ax=plt.subplots(111)
ax=plt.gca()
ax.set_xlim(0.01,10)
ax.set_xscale('log')
plt.xlabel('Firing rate(Hz)')
plt.ylabel('Probability density($Hz^{-1}$)')
sns.distplot(e_firing_rates,color='b')
plt.figure()
#ax=plt.subplots(111)
ax=plt.gca()
ax.set_xlim(0.1,100)
ax.set_xscale('log')
plt.xlabel('Firing rate(Hz)')
plt.ylabel('Probability density($Hz^{-1}$)')
sns.distplot(i_firing_rates,color='r')

n_sessions=6
plt.figure(figsize=(36,6))
for i in range(n_sessions):
    plt.subplot(1,2*n_sessions,2*i+1)
    e_neurons_sample=random.sample(e_firing_rates,20)
    #plt.bar(range(20),e_rates[i,0:20],color='b')
    plt.bar(x=0,bottom=range(20),width=e_neurons_sample,height=0.5, color='b',orientation="horizontal")
    plt.subplot(1,2*n_sessions,2*i+2)
    i_neurons_sample=random.sample(i_firing_rates,20)
    #plt.bar(range(20),i_rates[i,0:20],color='r')
    plt.bar(x=0,bottom=range(20),width=i_neurons_sample,height=0.5, color='r',orientation="horizontal")
    