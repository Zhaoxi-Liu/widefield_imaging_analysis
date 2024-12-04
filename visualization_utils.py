import matplotlib.pyplot as plt
import numpy as np

def find_peak(data, win_index=None, baseline=0, signal_deflect=1):
    '''
    find peak at axis 0
    '''
    if win_index == None:
        win_index = [0,data.shape[0]]
    if signal_deflect == 1:
        peak = np.max(data[win_index[0]:win_index[1]],axis=0) - baseline
    else:
        peak = baseline - np.min(data[win_index[0]:win_index[1]],axis=0)
    peak = peak * (peak > 0)
    return peak

def plot_hist_polar(data, window=[0, 10], title=None):

    xlim_min = 0
    xlim_max = data.shape[0]
    ylim_min = np.min(data)
    ylim_max = np.max(data)

    n_stim = data.shape[1]

    # for plot the directioin preference
    angles = np.arange(0, 361, 30)
    theta = np.deg2rad(angles)

    fig = plt.figure(layout='constrained', figsize=(8, 6))
    subfigs = fig.subfigures(1, 2, wspace=0.07)

    axis = subfigs[0].subplots(n_stim, 1)
    for i in range(n_stim):

        axis[i].bar(np.arange(data.shape[0]), data[:, i], width=1.0, color='k')
        axis[i].set_xlim(xlim_min, xlim_max)
        axis[i].set_ylim(ylim_min, ylim_max)
        axis[i].set_xticks([])
        # axis[i].set_yticks([])
        axis[i].spines['top'].set_visible(False)
        axis[i].spines['right'].set_visible(False)
        # axis[i].spines['left'].set_visible(False)
        axis[i].spines['bottom'].set_visible(False)
        axis[i].hlines(y=0, xmin=window[0], xmax=window[1], color='k', linewidth=1, linestyle='-') # 
        axis[i].set_ylabel('{}'.format(angles[i]))

    ax_polar = subfigs[1].add_subplot(projection='polar')
    
    response = np.mean(data, axis=0)
    response = np.append(response, response[0]) # append a element for plot a closed response
    ax_polar.plot(theta, response)
    ax_polar.set_theta_direction(-1)
    # ax_polar.tick_params(labelbottom=False)
    ax_polar.tick_params(labelleft=False)
    plt.suptitle(title)

def plot_response(data):
    [stim_len, n_stim, n_rep] = data.shape
    print(stim_len, n_stim, n_rep)
    xlim_max = data.shape[0]

    x = np.arange(xlim_max) * 0.1

    fig, axis = plt.subplots(n_stim, 1, figsize=(5, 10))
    for i in range(n_stim):
        mean = np.mean(data[:, i, :-1], axis=1)
        std = np.std(data[:, i, :-1], axis=1)
        y1 = mean + std
        y2 = mean - std
        axis[i].fill_between(x, y1, y2, alpha=0.3)
        axis[i].plot(x, mean)
        print(data[:, i, :-1].shape)
        axis[i].plot(x, data[:, i, :-1], alpha=0.5)

def plot_response_polar(data, window=[0, 10], title=None):

    xlim_min = 0
    xlim_max = data.shape[0]
    ylim_min = np.min(data)
    ylim_max = np.max(data)

    n_stim = data.shape[1]

    # for plot the directioin preference
    angles = np.arange(0, 361, 30)
    theta = np.deg2rad(angles)

    fig = plt.figure(layout='constrained', figsize=(8, 6))
    subfigs = fig.subfigures(1, 2, wspace=0.07)

    axis = subfigs[0].subplots(n_stim, 1)
    for i in range(n_stim):

        mean = np.mean(data[:,i,:], axis=1)
        y1 = mean + np.std(data[:,i,:], axis=1)
        y2 = mean - np.std(data[:,i,:], axis=1)
        axis[i].fill_between(np.arange(xlim_max), y1, y2, alpha=0.3)
        axis[i].plot(mean)
        axis[i].set_xlim(xlim_min, xlim_max)
        axis[i].set_ylim(ylim_min, ylim_max)
        axis[i].set_xticks([])
        axis[i].set_yticks([])
        axis[i].spines['top'].set_visible(False)
        axis[i].spines['right'].set_visible(False)
        axis[i].spines['left'].set_visible(False)
        axis[i].spines['bottom'].set_visible(False)
        axis[i].hlines(y=0, xmin=window[0], xmax=window[1], color='r', linewidth=1, linestyle='--')
        axis[i].set_ylabel('{}'.format(angles[i]))

    ax_polar = subfigs[1].add_subplot(projection='polar')
    
    response = find_peak(data=np.mean(data, axis=2))
    response = np.append(response, response[0]) # append a element for plot a closed response
    ax_polar.plot(theta, response)
    ax_polar.set_theta_direction(-1)
    # ax_polar.tick_params(labelbottom=False)
    ax_polar.tick_params(labelleft=False)
    plt.suptitle(title)