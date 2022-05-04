# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 16:26:38 2022

@author: alan
"""
from pfilter import ParticleFilter, independent_sample, squared_error
from scipy.stats import norm, gamma, uniform
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import pickle
import scipy.stats as st

np.random.seed(1)
# %%


# def apply_filter(y_train, pf, inputs=None):
#     states = []
#     pf.init_filter()  # reset
#     x_1 = {'x_1': uniform(0.9, 0.1).rvs()}  # intervalo do chute incial para x_1, precisa ser ajustado para cada caso
#     for i, tau in enumerate(inputs):
#         try:
#             pf.update(y_train[i], **x_1)
#             x_1 = {'x_1': y_train[i]}
#             print('mean state1', pf.mean_state, 'mean hypo1', pf.mean_hypothesis, 'std weigh', np.std(pf.weights))
#         except:
#             pf.update(None, **x_1)
#             x_1 = {'x_1': pf.mean_hypothesis}
#             print('mean state2', pf.mean_state, 'mean hypo2', pf.mean_hypothesis)

#         states.append([pf.hypotheses, np.array(pf.weights)])
#     return {
#         name: np.array([s[i] for s in states])
#         for i, name in enumerate(["particles", "c"])
#     }


def apply_filter(y_train, pf, inputs=None):
    states = []
    pf.init_filter()  # reset
    states.append([pf.particles[:, 0], np.array(pf.weights)])
    for i in range(1, len(inputs)):
        if i < len(y_train):
            pf.update(y_train[i])
            print('mean state1', pf.mean_state, 'mean hypo1', pf.mean_hypothesis, 'std weigh', np.std(pf.weights))
        else:
            pf.update(None)
            print('mean state2', pf.mean_state, 'mean hypo2', pf.mean_hypothesis)

        states.append([pf.particles[:, 0], np.array(pf.weights)])
    return {
        name: np.array([s[i] for s in states])
        for i, name in enumerate(["particles", "weights"])
    }


# def apply_filter(y_train, pf, inputs=None):
#     states = []
#     pf.init_filter()  # reset
#     for i in range(len(inputs)):

#         if i < len(y_train):
#             pf.update(y_train[i])
#             print('mean state1', pf.mean_state, 'mean hypo1', pf.mean_hypothesis, 'std weigh', np.std(pf.weights))
#             x_1 = pf.mean_hypothesis
#             states.append([pf.hypotheses, np.array(pf.weights)])
#         else:
#             resul = fun_obs(np.array([x_1, pf.mean_state[1]/time_int]).reshape(-1, 2))
#             states.append([np.repeat(resul, len(pf.weights)), np.ones(len(pf.weights))])
#             pf.update(None)
#             x_1 = resul.item()

#     return {
#         name: np.array([s[i] for s in states])
#         for i, name in enumerate(["particles", "weights"])
#     }


# %%
def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values.ravel(), weights=weights.ravel())
    # Fast and numerically precise:
    variance = np.average((values.ravel()-average)**2, weights=weights.ravel())
    return (average, np.sqrt(variance))
# %%


def plot_particles(x_train, x_test, y_train, y_test, states):
    fig, ax = plt.subplots()
    ax.scatter(x_train, y_train, label='Train', lw=1)
    # ax.scatter(x_test, y_test, label='Test', lw=1)

    particles = states["particles"]
    ws = states["weights"]

    stds_means = []
    for row in zip(particles, ws):
        stds_means.append(weighted_avg_and_std(row[0], row[1]))

    stds_means = np.array(stds_means)
    means = stds_means[:, 0]
    stds = stds_means[:, 1]

    ci = []
    for k in range(len(means)):
        ci.append(st.norm.interval(alpha=0.95, loc=means[k], scale=stds[k]))
    ci = np.array(ci)

    ax.plot(np.concatenate([x_train, x_test]), means, '*-', label='Mean est.')
    ax.plot(np.concatenate([x_train, x_test]), ci[:, 0], color='red', label='CI')
    ax.plot(np.concatenate([x_train, x_test]), ci[:, 1], color='red', label='CI')

    ax.fill_between(np.concatenate([x_train, x_test]).ravel(), means-stds, means+stds,
                    color='C4', alpha=0.5, label='Std.')
    # ax.scatter(np.tile(np.concatenate([x_train, x_test]), (len(particles[0]), 1)).ravel(), particles.T, s=ws*10000/np.sqrt(len(ws)),
    #            alpha=0.15, label='Hypotheses')
    ax.set_xlabel("Time")
    ax.set_ylabel("Observed")
    ax.legend()
    return means

# %%


def filter_plot(x_train, x_test, y_train, y_test, pf, inputs=None):
    states = apply_filter(y_train, pf, inputs)
    kau = plot_particles(x_train, x_test, y_train, y_test, states)
    return kau
# %%


# with open('train.pickle', 'rb') as f:
#     train = pickle.load(f)

# with open('test.pickle', 'rb') as f:
#     test = pickle.load(f)

# engine = 0

# y_train = np.array(test[engine][:70])
# y_test = np.array(test[engine][70:])


# time_int = 1
# time = np.zeros((len(test[engine]), 1))
# time[0, :] = 0
# for i in range(1, len(test[engine])):
#     time[i, :] = time[i-1, :]+time_int


# x_train = time[:70]
# x_test = time[70:]

# prior_fn = independent_sample([uniform(0, 0.02).rvs,
#                                uniform(0.0, 1.5).rvs])

# noise = 0.005

# sigma = 0.00001


# def fun_obs(x):
#     return np.exp(-1*x[:, 1]*time_int)*x[:, 0]


# def din_obs(x):
#     return np.array([np.exp(x[:, 1])*x[:, 0], x[:, 1]]).T


# def noise_func(x):
#     x = x + np.random.normal(0, noise, x.shape)
#     return x


# %% DOIS STATES
# data = sio.loadmat('TESTLED10.mat')['led']
# split = 35

# # scaler = MinMaxScaler()
# # data_scaled = scaler.fit_transform(data)
# data_scaled = data

# y_train = data_scaled[:split]
# y_test = data_scaled[split:]

# time_int = 50
# time = np.zeros((len(data), 1))
# time[0, :] = 0
# for i in range(1, len(data)):
#     time[i, :] = time[i-1, :]+time_int

# x_train = time[:split]
# x_test = time[split:]

# prior_fn = independent_sample([uniform(90, 20).rvs,
#                                uniform(0, 1).rvs])

# sigma = np.var(y_train)
# noise = 0.00


# def fun_obs(x):
#     return np.exp(-1*x[:, 1])*x[:, 0]


# def din_obs(x):
#     return np.array([np.exp(-1*x[:, 1])*x[:, 0], x[:, 1]]).T


# def weight(x, y, **kwargs):
#     dx = (x - y) ** 2
#     d = np.sum(dx, axis=1)
#     return (1/np.sqrt((2*np.pi*sigma))) * np.exp(-d / (2.0 * sigma)) + 1e-99


# def noise_func(x):
#     x = x + np.random.normal(0, noise, x.shape)
#     return x

# %% NORMAL

# measuData = np.array([1.0, 0.9351, 0.8512, 0.9028, 0.7754, 0.7114, 0.6830, 0.6147, 0.5628, 0.7090])


# time_int = 5
# time = np.zeros((22, 1))
# time[0, :] = 0
# for i in range(1, 22):
#     time[i, :] = time[i-1, :]+time_int

# y_train = measuData[:10]
# # y_test = measuData[10:]
# y_test = np.ones(12)*0.5
# x_train = time[:10]
# x_test = time[10:]


# prior_fn = independent_sample([uniform(0, 0.05).rvs])

# noise = 0.00
# sigma = 0.01


# def fun_obs(x, x_1):
#     x = np.exp(-1*x[:, 0]*time_int)*x_1
#     return x


# def weight(x, y, **kwargs):
#     dx = (x - y) ** 2
#     d = np.sum(dx, axis=1)
#     return (1/np.sqrt((2*np.pi*sigma))) * np.exp(-d / (2.0 * sigma))
# %% COM DUAS PARTICULAS


# measuData = np.array([1.0, 0.9351, 0.8512, 0.9028, 0.7754, 0.7114, 0.6830, 0.6147, 0.5628, 0.7090])


# time_int = 1
# time = np.zeros((22, 1))
# time[0, :] = 0
# for i in range(1, 22):
#     time[i, :] = time[i-1, :]+time_int

# y_train = measuData[:10]
# # y_test = measuData[10:]
# y_test = np.ones(12)*0.5
# x_train = time[:10]
# x_test = time[10:]


# prior_fn = independent_sample([uniform(0.9, 0.2).rvs,
#                                uniform(0, 0.05).rvs])

# sigma = np.var(measuData)
# noise = 0.00


# def fun_obs(x):
#     return np.exp(-1*x[:, 1])*x[:, 0]


# def din_obs(x):
#     return np.array([np.exp(-1*x[:, 1])*x[:, 0], x[:, 1]]).T


# def noise_func(x):
#     x = x + np.random.normal(0, noise, x.shape)
#     return x


# def weight(x, y, **kwargs):
#     dx = (x - y) ** 2
#     d = np.sum(dx, axis=1)
#     return (1/np.sqrt((2*np.pi*sigma))) * np.exp(-d / (2.0 * sigma))


# %%

data = pd.read_csv('fc1.csv', index_col=0)
data.index = pd.to_timedelta(data.index, unit='hour')
# data = data.resample('30min').mean()

y_train = data['Utot (V)'][data.index <= '550h'].values
y_test = data['Utot (V)'][data.index > '550h'].values

x_train = np.array(data['Utot (V)'][data.index <= '550h'].index.total_seconds()/60)
x_test = np.array(data['Utot (V)'][data.index > '550h'].index.total_seconds()/60)

prior_fn = independent_sample([uniform(y_train[0]*0.9, y_train[0]*0.2).rvs,
                               uniform(0, 0.05).rvs,
                               uniform(0, 0.05).rvs])

sigma = np.var(y_train)
noise = 0.00


def fun_obs(x, time):
    return -1*x[:, 1]*time**2 - 1*x[:, 2]*time+x[:, 0]


def din_obs(x, time):
    return np.array([fun_obs(x_time), x[:, 1]]).T


def noise_func(x):
    x = x + np.random.normal(0, noise, x.shape)
    return x


def weight(x, y, **kwargs):
    dx = (x - y) ** 2
    d = np.sum(dx, axis=1)
    return (1/np.sqrt((2*np.pi*sigma))) * np.exp(-d / (2.0 * sigma))

# %%
# measuData = np.array([0.0119, 0.0103, 0.0118, 0.0095, 0.0085,
#                       0.0122, 0.0110, 0.0120, 0.0113, 0.0122, 0.0110, 0.0124, 0.0117,
#                       0.0138, 0.0127, 0.0115, 0.0135, 0.0124, 0.0141, 0.0160, 0.0157,
#                       0.0149, 0.0156, 0.0153, 0.0155])


# time_int = 50
# time = np.zeros((51, 1))
# time[0, :] = 0
# for i in range(1, 51):
#     time[i, :] = time[i-1, :]+time_int

# y_train = measuData[:25]
# # y_test = measuData[10:]
# y_test = np.ones(26)*0.0110
# x_train = time[:25]
# x_test = time[25:]

# prior_fn = independent_sample([uniform(5e-4, 0.0095).rvs,
#                                uniform(-22.33, 23.53).rvs,
#                                uniform(0.2, 3.8).rvs])
# noise = 0.00
# sigma = 3.918816e-06


# def fun_obs(x):
#     return np.exp(x[:, 1])*(78*np.sqrt(np.pi*x[:, 0]))**x[:, 2] + x[:, 0]


# def din_obs(x):
#     nex_x = np.exp(x[:, 1])*(78*np.sqrt(np.pi*x[:, 0]))**x[:, 2] + x[:, 0]
#     return np.array([nex_x, x[:, 1], x[:, 2]]).T


# def weight(x, y, **kwargs):
#     dx = (x - y) ** 2
#     d = np.sum(dx, axis=1)
#     return (1/np.sqrt((2*np.pi*sigma))) * np.exp(-d / (2.0 * sigma))


# def noise_func(x):
#     x = x + np.random.normal(0, noise, x.shape)
#     return x


# %%
ts = [{"t": t} for t in np.concatenate([x_train, x_test])]

pf = ParticleFilter(prior_fn=prior_fn,
                    observe_fn=fun_obs,
                    n_particles=5000,
                    dynamics_fn=din_obs,
                    n_eff_threshold=0.5,
                    noise_fn=noise_func,
                    # resample_proportion=0.02,
                    weight_fn=weight
                    )
# %%
% matplotlib inline
theta = filter_plot(x_train, x_test, y_train, y_test, pf, inputs=ts)
plt.grid()
plt.xlim([0, 20])
plt.ylim([0, 1.2])
