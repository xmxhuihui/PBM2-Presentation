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
W_EE2 = 0.26

mu_EI = W_EI
mu_IE = W_IE
mu_II = W_II
mu_EE = W_EE
sigma_EI2 = W_EI2 - W_EI ** 2
sigma_IE2 = W_IE2 - W_IE2 ** 2
sigma_II2 = W_II2 - W_II ** 2
sigma_EE2 = W_EE2 - W_EE ** 2
sigma_EI = np.sqrt(sigma_EI2)
sigma_IE = np.sqrt(sigma_IE2)
sigma_II = np.sqrt(sigma_II2)
sigma_EE = np.sqrt(sigma_EE2)

mu_EI_ln = np.log(mu_EI**2/np.sqrt(mu_EI**2+sigma_EI2))
sigma_EI_ln = np.sqrt(np.log(1 + sigma_EI2/mu_EI**2))
mu_IE_ln = np.log(mu_IE**2/np.sqrt(mu_IE**2+sigma_IE2))
sigma_IE_ln = np.sqrt(np.log(1 + sigma_IE2/mu_IE**2))
mu_II_ln = np.log(mu_II**2/np.sqrt(mu_II**2 + sigma_II2))
sigma_II_ln = np.sqrt(np.log(1 + sigma_II2/mu_II**2))
mu_EE_ln = np.log(mu_EE**2/np.sqrt(mu_EE**2 + sigma_EE2))
sigma_EE_ln = np.sqrt(np.log(1 + sigma_EE2/mu_EE**2))

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

    for i in range(n_neurons):
        for j in range(n_neurons):
            if i < N_E:
                # E -> E
                if j < N_E:
                    if random.uniform(0, 1) <= c_EE:
                        # index = random.randint(1, 3688)
                        # W[i, j] = spines_info['Volume'].loc[spines_info['Global_SpineID'] == index].values[0] * g
                        W[i, j] = np.random.lognormal(mu_EE_ln, sigma_EE_ln)
                # E -> I
                else:
                    if random.uniform(0, 1) <= c_EI:
                        W[i, j] = -np.random.lognormal(mu_EI_ln, sigma_EI_ln)
            else:
                # I -> E
                if j < N_E:
                    if random.uniform(0, 1) <= c_IE:
                        W[i, j] = np.random.lognormal(mu_IE_ln, sigma_IE_ln)
                # I -> I
                else:
                    if random.uniform(0, 1) <= c_II:
                        W[i, j] = -np.random.lognormal(mu_II_ln, sigma_II_ln)
    return W


W = np.zeros((n_sessions, n_neurons, n_neurons))
plt.figure(figsize=(24, 4))
for i in range(n_sessions):
    W[i] = W_Construction(i + 1)
    # print('Session {}'.format(i + 1))
    # w_ee = np.array(W[i, 0:N_E, 0:N_E]).flatten()
    # w_ei = np.array(W[i, 0:N_E, N_E:]).flatten()
    # w_ie = np.array(W[i, N_E:, 0:N_E]).flatten()
    # w_ii = np.array(W[i, N_E:, N_E:]).flatten()
    # print('###### E -> E #######')
    # print('Mean: {}'.format(np.mean(w_ee)))
    # print('Variance: {}'.format(np.var(w_ee)))
    # print('###### E -> I #######')
    # print('Mean: {}'.format(np.mean(w_ei)))
    # print('Variance: {}'.format(np.var(w_ei)))
    # print('###### I -> E #######')
    # print('Mean: {}'.format(np.mean(w_ie)))
    # print('Variance: {}'.format(np.var(w_ie)))
    # print('###### I -> I #######')
    # print('Mean: {}'.format(np.mean(w_ii)))
    # print('Variance: {}'.format(np.var(w_ii)))
    plt.subplot(1, 6, i + 1)
    sns.heatmap(W[i], vmin=0, vmax=1.6, cmap='jet')


def sess(IS):
    # Recording the state of each neuron in the last timestep
    for i in range(n_neurons):
        v[i, 0] = v_R
    t = range(total_time - 1)

    # For excitatory neurons
    for dt in t:

        for i in range(n_neurons):
            for j in range(n_neurons):
                h[i, dt] = h[i, dt] + W[IS - 1, i, j] * r[j]
            if v[i, dt] == spike:
                v[i, dt + 1] = v_R
            else:
                if i < N_E:
                    v[i, dt + 1] = v[i, dt] - v[i, dt] / tau_m + h[i, dt] + H_E / tau_m
                else:
                    v[i, dt + 1] = v[i, dt] - v[i, dt] / tau_m + h[i, dt] + H_I / tau_m

                if v[i, dt + 1] >= theta:
                    v[i, dt + 1] = spike
                    r[i] = 1
                    if i < N_E:
                        e_firing_time[i].append(dt + 1)
                    else:
                        i_firing_time[i - N_E].append(dt + 1)
                else:
                    r[i] = 0


# Method 2:
e_firing_rates = [[] for i in range(n_sessions)]
i_firing_rates = [[] for i in range(n_sessions)]
e_firing_rates_2 = []
i_firing_rates_2 = []
duration = 50
for i in range(n_sessions):
    v = np.zeros((n_neurons, total_time))
    h = np.zeros((n_neurons, total_time))
    r = np.zeros(n_neurons)
    e_firing_time = [[] for i in range(N_E)]
    i_firing_time = [[] for i in range(N_I)]
    sess(i + 1)
    for l in range(n_neurons):
        for j in range(int(total_time/duration)):
            e_spikes_2 = 0
            i_spikes_2 = 0
            if l < N_E:
                for k in e_firing_time[l]:
                    if duration * j <= k < duration * (j + 1):
                        e_spikes_2 += 1
            else:
                for k in i_firing_time[l - N_E]:
                    if duration * j <= k < duration * (j + 1):
                        i_spikes_2 += 1
            if e_spikes_2 != 0:
                e_firing_rates[i].append(e_spikes_2 / duration * 1000)
                e_firing_rates_2.append(e_spikes_2 / duration * 1000)
            if i_spikes_2 != 0:
                i_firing_rates[i].append(i_spikes_2 / duration * 1000)
                i_firing_rates_2.append(i_spikes_2 / duration * 1000)


plt.figure(figsize=(48, 8))
for i in range(n_sessions):
    plt.subplot(1, 2 * n_sessions, 2 * i + 1)
    e_neurons_sample = random.sample(e_firing_rates[i], 20)
    # plt.bar(range(20),e_rates[i,0:20],color='b')
    plt.bar(x=0, bottom=range(20), width=e_neurons_sample, height=0.5, color='b', orientation="horizontal")
    plt.subplot(1, 2 * n_sessions, 2 * i + 2)
    i_neurons_sample = random.sample(i_firing_rates[i], 20)
    # plt.bar(range(20),i_rates[i,0:20],color='r')
    plt.bar(x=0, bottom=range(20), width=i_neurons_sample, height=0.5, color='r', orientation="horizontal")

# Plot the spiking of an excitatory and inhibitory neuron
plt.figure()
plt.plot(range(total_time), v[0], 'b')
plt.show()
plt.figure()
plt.plot(range(total_time), v[N_E], 'r')
plt.show()

plt.figure()
plt.xlim(0, max(e_firing_rates[5]))
sns.distplot(e_firing_rates_2, color='b')
plt.show()
plt.figure()
plt.xlim(0, max(i_firing_rates[5]))
sns.distplot(i_firing_rates_2, color='r')
plt.show()

e_firing_rates_log_2 = np.log(e_firing_rates_2)
i_firing_rates_log_2 = np.log(i_firing_rates_2)
plt.figure()
plt.xlim(min(e_firing_rates_log_2), max(e_firing_rates_log_2))
sns.distplot(e_firing_rates_log_2, color='b')
plt.show()

plt.figure()
plt.xlim(min(i_firing_rates_log_2), max(i_firing_rates_log_2))
sns.distplot(i_firing_rates_log_2, color='r')
plt.show()
# plot_acf(W[0:6,0:N_E,0:N_E])

# Autocorrelogram
e_corr_scores = []
i_corr_scores = []

# Autocorrelogram scores for excitatory and inhibitory firing rates
e_s1 = pd.Series(e_firing_rates[0])
i_s1 = pd.Series(i_firing_rates[0])
for i in range(n_sessions):
    e_s2 = pd.Series(e_firing_rates[i])
    e_corr_scores.append(e_s1.corr(e_s2))

    i_s2 = pd.Series(i_firing_rates[i])
    i_corr_scores.append((i_s1.corr(i_s2)))

print(e_corr_scores)
print(i_corr_scores)

# Autocorrelogram scores for E->E connectivity matrix
arr = np.array(W[0, 0:N_E, 0:N_E])
s1 = pd.Series(arr.flatten())
e_e_corr_Scores = []
for i in range(n_sessions):
    s2 = pd.Series(np.array(W[i, 0:N_E, 0:N_E]).flatten())
    e_e_corr_Scores.append(s1.corr(s2))

print(e_e_corr_Scores)
