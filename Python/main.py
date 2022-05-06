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

np.random.seed(1)
# %%


def apply_filter(y_train, pf, inputs=None):
    states = []
    pf.init_filter()  # reset
    for i, tau in enumerate(inputs):
        try:
            pf.update(y_train[i])
            print('mean state1', pf.mean_state, 'mean hypo1', pf.mean_hypothesis, 'std weigh', np.std(pf.weights))
        except:
            pf.update(None)
            print('mean state2', pf.mean_state, 'mean hypo2', pf.mean_hypothesis)

        states.append([pf.particles[:, 0], np.array(pf.weights)])
    return {
        name: np.array([s[i] for s in states])
        for i, name in enumerate(["particles", "weights"])
    }


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
    ax.scatter(x_test, y_test, label='Test', lw=1)

    particles = states["particles"]
    ws = states["weights"]

    stds_means = []
    for row in zip(particles, ws):
        stds_means.append(weighted_avg_and_std(row[0], row[1]))

    stds_means = np.array(stds_means)
    means = stds_means[:, 0]
    stds = stds_means[:, 1]

    ax.plot(np.concatenate([x_train, x_test]), means, '*-', label='Mean est.')

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


measuData = np.array([1.0, 0.9351, 0.8512, 0.9028, 0.7754, 0.7114, 0.6830, 0.6147, 0.5628, 0.7090])


time_int = 5
time = np.zeros((22, 1))
time[0, :] = 0
for i in range(1, 22):
    time[i, :] = time[i-1, :]+time_int

y_train = measuData[:10]
# y_test = measuData[10:]
y_test = np.ones(12)*0.5
x_train = time[:10]
x_test = time[10:]


prior_fn = independent_sample([uniform(0.9, 0.3).rvs,
                               uniform(0, 0.05).rvs])

sigma = 0.01
noise = 0.005


def fun_obs(x):
    return np.exp(-1*x[:, 1])*x[:, 0]


def din_obs(x):
    return np.array([np.exp(-1*x[:, 1])*x[:, 0], x[:, 1]]).T


def noise_func(x):
    x = x + np.random.normal(0, noise, x.shape)
    return x


def weight(x, y, **kwargs):
    dx = (x - y) ** 2
    d = np.sum(dx, axis=1)
    return (1/np.sqrt((2*np.pi*sigma))) * np.exp(-d / (2.0 * sigma))


# %%

ts = [{"t": t} for t in np.concatenate([x_train, x_test])]

pf = ParticleFilter(prior_fn=prior_fn,
                    observe_fn=fun_obs,
                    n_particles=6500,
                    dynamics_fn=din_obs,
                    n_eff_threshold=0.5,
                    noise_fn=noise_func,
                    # resample_proportion=0.02,
                    weight_fn=weight
                    )
# %%
%matplotlib inline
theta = filter_plot(x_train, x_test, y_train, y_test, pf, inputs=ts)
plt.grid()
