import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf


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
print(str(sigma_EI2) + '\n' + str(sigma_IE2) + '\n' + str(sigma_II2))

theta = 33
tau_m = 10
H_E = 77.6
H_I = 57.8
v_R = 24.75
spike = 150

e_rates = np.zeros((n_sessions,N_E))
i_rates = np.zeros((n_sessions,N_I))
e_firing_rates = []
i_firing_rates = []
# Extracting E->E connectivity from the spine imaging data
c_EE = 0.2
path = "Global_Spines_info.csv"
spines_info = pd.read_csv(path)
spines_info.drop('Unnamed: 0', axis=1, inplace=True)

spines_IS1 = spines_info.loc[spines_info['Starting Imaging Session'] == 1]
# spines_IS1.head(100)
S = spines_IS1['Volume'].mean()
g = W_EE / S
print(g)


# Connectivity matrix 8*8
# EI
# I*
def W_Construction(IS):
    c_EE = 0.2
    c_EI = 0.4
    c_IE = 0.3
    c_II = 0.4
    W = np.zeros((n_neurons, n_neurons))
    c = np.zeros((n_neurons, n_neurons))

    # E->I connections
    for i in range(N_E):
        for j in range(N_E, n_neurons):
            if random.uniform(0, 1) <= c_EI:
                c[i, j] = 1
                W[i, j] = np.random.lognormal(mu_EI, sigma_EI)

    # I->E connections
    for i in range(N_E, n_neurons):
        for j in range(N_E):
            if random.uniform(0, 1) <= c_IE:
                c[i, j] = 1
                W[i, j] = np.random.lognormal(mu_IE, sigma_IE)

    # I->I connections
    for i in range(N_E, n_neurons):
        for j in range(N_E, n_neurons):
            if random.uniform(0, 1) <= c_II:
                c[i, j] = 1
                W[i, j] = np.random.lognormal(mu_II, sigma_II)

    # E->E connections
    #index_list = spines_info['Global_SpineID'].loc[(spines_info['Starting Imaging Session']<=IS)&(spines_info['Ending Imaging Session']>=IS)]
    #print(len(index_list))
    for i in range(N_E):
        for j in range(N_E):
            if random.uniform(0, 1) <= c_EE:
                index = random.randint(1,3688)
                W[i, j] = spines_info['Volume'].loc[spines_info['Global_SpineID'] == index].values[0] * g
                c[i, j] = 1
            else:
                W[i, j] = 0
    return W, c


W = np.zeros((n_sessions, n_neurons, n_neurons))
c = np.zeros((n_sessions, n_neurons, n_neurons))
plt.figure(figsize=(24, 4))
for i in range(n_sessions):
    W[i], c[i] = W_Construction(i+1)
    plt.subplot(1, 6, i + 1)
    sns.heatmap(W[i], vmin=0, vmax=1.6, cmap='jet')
# Recurrent input of neuron i

# for i in range(n_neurons):
#     for j in range(N_E):
#         tsteps = 0
#         while (True):
#             tstep = random.randint(10, 50)
#             tsteps = tsteps + tstep
#             if tsteps < total_time:
#                 h[i, tsteps] = h[i, tsteps] + c[0, i, j] * W[0, i, j]
#             else:
#                 break
#     for j in range(N_E, n_neurons):
#         tsteps = 0
#         while (True):
#             tstep = random.randint(10, 50)
#             tsteps = tsteps + tstep
#             if tsteps < total_time:
#                 h[i, tsteps] = h[i, tsteps] - c[0, i, j] * W[0, i, j]
#             else:
#                 break


# Potential of neuron




def sess(IS):
  # Recording the state of each neuron in the last timestep
    for i in range(n_neurons):
        v[i, 0] = v_R
    t = range(total_time - 1)
    for dt in t:
        e_spikes = 0
        i_spikes = 0
        # For excitatory neurons
        for i in range(N_E):
            for j in range(N_E):
                h[i, dt] = h[i, dt] + c[IS-1, i, j] * W[IS-1, i, j] * r[j]
            for j in range(N_E, n_neurons):
                h[i, dt] = h[i, dt] - c[IS-1, i, j] * W[IS-1, i, j] * r[j]
            if v[i, dt] == spike:
                v[i, dt + 1] = v_R
            else:
                v[i, dt + 1] = v[i, dt] - v[i, dt] / tau_m + h[i, dt] + H_E / tau_m
                if v[i, dt + 1] >= theta:
                    v[i, dt + 1] = spike
                    r[i] = 1
                    e_spikes += 1
                    e_rates[IS-1,i]=e_rates[IS-1,i]+1
                else:
                    r[i] = 0
    
        firing_rate = e_spikes / N_E * 1000
        e_firing_rates.append(firing_rate)
        # For inhibitory neurons
        for i in range(N_E, n_neurons):
            for j in range(N_E):
                h[i, dt] = h[i, dt] + c[0, i, j] * W[0, i, j] * r[j]
            for j in range(N_E, n_neurons):
                h[i, dt] = h[i, dt] - c[0, i, j] * W[0, i, j] * r[j]
            if v[i, dt] == spike:
                v[i, dt + 1] = v_R
            else:
                v[i, dt + 1] = v[i, dt] - v[i, dt] / tau_m + h[i, dt] + H_I / tau_m
                if v[i, dt + 1] >= theta:
                    v[i, dt + 1] = spike
                    r[i] = 1
                    i_spikes += 1
                    i_rates[IS-1,i-N_E]=i_rates[IS-1,i-N_E]+1
                else:
                    r[i] = 0
    
        firing_rate = i_spikes / N_I * 1000
        i_firing_rates.append(firing_rate)


for i in range(n_sessions):
    v = np.zeros((n_neurons, total_time))
    h = np.zeros((n_neurons, total_time))
    r = np.zeros(n_neurons)
    sess(i+1)
e_rates=e_rates/total_time*1000
i_rates=i_rates/total_time*1000

plt.figure(figsize=(48,8))
for i in range(n_sessions):
    plt.subplot(1,2*n_sessions,2*i+1)
    #plt.bar(range(20),e_rates[i,0:20],color='b')
    plt.bar(x=0,bottom=range(20),width=e_rates[i,0:20],height=0.5, color='b',orientation="horizontal")
    plt.subplot(1,2*n_sessions,2*i+2)
    #plt.bar(range(20),i_rates[i,0:20],color='r')
    plt.bar(x=0,bottom=range(20),width=i_rates[i,0:20],height=0.5, color='r',orientation="horizontal")
# Plot the spiking of an excitatory and inhibitory neuron
# plt.figure()
# plt.plot(range(total_time), v[0], 'b')
# plt.show()
# plt.figure()
# plt.plot(range(total_time), v[N_E], 'r')
# plt.show()
plt.figure()
plt.xlim(0, 400)
sns.distplot(e_firing_rates, bins=80, color='b')
plt.show()
plt.figure()
plt.xlim(0, 800)
sns.distplot(i_firing_rates, bins=80, color='r')
plt.show()

e_firing_rates_log = []
i_firing_rates_log = []
plt.figure()
for i in e_firing_rates:
    e_firing_rates_log.append(np.log10(i + 1))
plt.xlim(min(e_firing_rates_log), max(e_firing_rates_log))
sns.distplot(e_firing_rates_log, color='b')
plt.show()
plt.figure()
for i in i_firing_rates:
    i_firing_rates_log.append(np.log10(i + 1))
plt.xlim(min(i_firing_rates_log), max(i_firing_rates_log))
sns.distplot(i_firing_rates_log, color='r')
plt.show()
# for i in range(n_neurons):
#     arr, count = np.unique(v[i], return_counts=True)
#     frequency = dict(zip(arr, count))
#     if i < N_E:
#         e_rates.append(frequency[spike] / total_time * 1000)
#     else:
#         i_rates.append(frequency[spike] / total_time * 1000)

# plt.figure()
# plt.xlim(0, max(e_rates))
# plt.hist(e_rates, bins=90, density=True)
# plt.show()
# plt.figure()
# plt.xlim(0, max(i_rates))
# plt.hist(i_rates, bins=90, density=True)
# plt.show()
# plt.figure()
# e_rates_log = np.log(e_rates)
# plt.xlim(min(e_rates_log), max(e_rates_log))
# plt.hist(e_rates_log, bins=90, density=True)
# plt.show()
# plt.figure()
# i_rates_log = np.log(i_rates)
# plt.xlim(min(i_rates_log), max(i_rates_log))
# plt.hist(i_rates_log, bins=90, density=True)
# plt.show()

# plot_acf(W[0:6,0:N_E,0:N_E])