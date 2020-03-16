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
total_time = 3000

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

# Extracting E->E connectivity from the spine imaging data
c_EE = 0.2
path = "Global_Spines_info.csv"
spines_info = pd.read_csv(path)
spines_info.drop('Unnamed: 0', axis=1, inplace=True)
spines_info.head()

spines_IS1 = spines_info.loc[spines_info['Starting Imaging Session'] == 1]
spines_IS1.head(100)  # Why 100 here?
S = spines_IS1['Volume'].mean()
g = W_EE / S
print(g)


# Connectivity matrix 8*8
# EI
# I*
def W_Construction():
    c_EE = 0.51
    W = np.zeros((n_neurons, n_neurons))
    c = np.zeros((n_neurons, n_neurons))

    # E->I connections
    for i in range(N_E):
        for j in range(N_E, n_neurons):
            if random.uniform(0, 1) <= 0.5:
                c[i, j] = 1
                W[i, j] = np.random.lognormal(mu_EI, sigma_EI)

    # I->E connections
    for i in range(N_E, n_neurons):
        for j in range(N_E):
            if random.uniform(0, 1) <= 0.5:
                c[i, j] = 1
                W[i, j] = np.random.lognormal(mu_IE, sigma_IE)

    # I->I connections
    for i in range(N_E, n_neurons):
        for j in range(N_E, n_neurons):
            if random.uniform(0, 1) <= 0.5:
                c[i, j] = 1
                W[i, j] = np.random.lognormal(mu_II, sigma_II)

    # E->E connections
    for i in range(N_E):
        for j in range(N_E):
            if random.uniform(0, 1) <= c_EE:
                index = random.randint(1, 3688)
                W[i, j] = spines_info['Volume'].loc[spines_info['Global_SpineID'] == index] * g
            else:
                W[i, j] = 0
            c[i, j] = c_EE
    return W, c


W = np.zeros((n_sessions, n_neurons, n_neurons))
c = np.zeros((n_sessions, n_neurons, n_neurons))
plt.figure(figsize=(24, 4))
for i in range(n_sessions):
    W[i], c[i] = W_Construction()
    plt.subplot(1, 6, i + 1)
    sns.heatmap(W[i], vmin=0, vmax=1.6, cmap='jet')
    # plt.show()

# Recurrent input of neuron i
h = np.zeros((n_neurons, total_time))
for i in range(n_neurons):
    for j in range(N_E):
        tsteps = 0
        while (True):
            tstep = random.randint(10, 50)
            tsteps = tsteps + tstep
            if tsteps < total_time:
                h[i, tsteps] = h[i, tsteps] + c[0, i, j] * W[0, i, j] * tstep
            else:
                break
    for j in range(N_E, n_neurons):
        tsteps = 0
        while (True):
            tstep = random.randint(10, 50)
            tsteps = tsteps + tstep
            if tsteps < total_time:
                h[i, tsteps] = h[i, tsteps] - c[0, i, j] * W[0, i, j] * tstep
            else:
                break

# Potential of neuron
v = np.zeros((n_neurons, total_time))
theta = 33
tau_m = 10
H_E = 77.6
H_I = 57.8
v_R = 24.75
for i in range(n_neurons):
    v[i, 0] = v_R
t = range(total_time - 1)
for dt in t:
    for i in range(N_E):
        if v[i, dt] == 1000:
            v[i, dt + 1] = v_R
        else:
            v[i, dt + 1] = v[i, dt] - v[i, dt] / tau_m + h[i, dt] + H_E / tau_m
            if v[i, dt + 1] >= theta:
                v[i, dt + 1] = 1000
    for i in range(N_E, n_neurons):
        if v[i, dt] == 1000:
            v[i, dt + 1] = v_R
        else:
            v[i, dt + 1] = v[i, dt] - v[i, dt] / tau_m + h[i, dt] + H_I / tau_m
            if v[i, dt + 1] >= theta:
                v[i, dt + 1] = 1000
plt.figure()
plt.plot(range(total_time), v[1])
plt.show()
plt.figure()
plt.plot(range(total_time), v[6], 'r')
plt.show()
