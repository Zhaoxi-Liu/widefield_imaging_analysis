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

def plot_dff(data, data_rate=None, figsize=(15, 5), title=None,
    xlim=None, ylim=None, labels=None, **kwargs):
    '''
    Plot the traces of the data.
    data: np.array, shape=(nframes, ntrials)
    data_rate: int, the rate of the data, default is None
    labels: None or list of str, the labels of the traces
    step: used to separate the dff traces
    '''
    fontsize = np.min(figsize) * 2.5
    # plt.rc('axes', labelsize=fontsize)
    plt.rc('xtick', labelsize=fontsize)
    plt.rc('ytick', labelsize=fontsize)

    if 'colors' not in kwargs:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig, ax = plt.subplots(figsize=figsize)

    if data_rate is not None:
        x = np.arange(data.shape[0]) / data_rate
    else:
        x = np.arange(data.shape[0])
    
    step = kwargs.get('step', 0.2)
    for i in range(data.shape[1]):
        if labels is not None:
            ax.plot(x, data[:, i]+i*step, label=labels[i], lw=1,
                color=colors[i])
        else:
            ax.plot(x, data[:, i]+i*step, lw=1, color=colors[i])

    if data_rate is not None:
        ax.set_xlabel('Time (s)')
    else:
        if 'xlabel' in kwargs:
            ax.set_xlabel(kwargs['xlabel'])

    if 'vlines' in kwargs and data_rate is not None:
        for vline in kwargs['vlines']:
            ax.axvline(x=vline, color='r', linestyle='--')

    if 'hlines' in kwargs:
        for i, hline in enumerate(kwargs['hlines']):
            ax.axhline(y=hline, color=colors[i])

    if title is not None:
        ax.set_title(title)
    if xlim is not None:
        ax.set_xlim(left=xlim[0], right=xlim[1])
    else:
        ax.set_xlim(left=-x[-1]*0.01, right=x[-1]*1.01)
    if ylim is not None:
        ax.set_ylim(bottom=ylim[0], top=ylim[1])
    if labels is not None:
        ax.legend()
    if 'ylabel' in kwargs:
        ax.set_ylabel(kwargs['ylabel'])

    plt.show()

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

def plot_traces(data, data_rate=None, figsize=(15, 5), title=None,
    xlim=None, ylim=None, labels=None, **kwargs):
    '''
    Plot the traces of the data.
    data: np.array, shape=(nframes, ntrials)
    data_rate: int, the rate of the data, default is None
    labels: None or list of str, the labels of the traces
    '''
    fontsize = np.min(figsize) * 2.5
    # plt.rc('axes', labelsize=fontsize)
    plt.rc('xtick', labelsize=fontsize)
    plt.rc('ytick', labelsize=fontsize)

    if 'colors' not in kwargs:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig, ax = plt.subplots(figsize=figsize)

    if data_rate is not None:
        x = np.arange(data.shape[0]) / data_rate
    else:
        x = np.arange(data.shape[0])

    for i in range(data.shape[1]):
        if labels is not None:
            ax.plot(x, data[:, i], label=labels[i], lw=1, color=colors[i])
        else:
            ax.plot(x, data[:, i], lw=1, color=colors[i])

    if data_rate is not None:
        ax.set_xlabel('Time (s)')
    else:
        if 'xlabel' in kwargs:
            ax.set_xlabel(kwargs['xlabel'])

    if 'vlines' in kwargs and data_rate is not None:
        for vline in kwargs['vlines']:
            ax.axvline(x=vline, color='r', linestyle='--')

    if 'hlines' in kwargs:
        for i, hline in enumerate(kwargs['hlines']):
            ax.axhline(y=hline, color=colors[i])

    if title is not None:
        ax.set_title(title)
    if xlim is not None:
        ax.set_xlim(left=xlim[0], right=xlim[1])
    else:
        ax.set_xlim(left=-x[-1]*0.01, right=x[-1]*1.01)
    if ylim is not None:
        ax.set_ylim(bottom=ylim[0], top=ylim[1])
    if labels is not None:
        ax.legend()
    if 'ylabel' in kwargs:
        ax.set_ylabel(kwargs['ylabel'])
    plt.show()

def show_array_images(images, n_cols=5, cmap='gray', **kwargs):
    '''
    Show a list of images in a grid.
    n_cols: number of columns in the grid.
    '''
    n_rows = int(np.ceil(len(images) / n_cols))
    figsize = (n_cols * 3, n_rows * 3)
    fontsize = np.min(figsize) * 2
    plt.rc('xtick', labelsize=fontsize)
    plt.rc('ytick', labelsize=fontsize)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize,
        layout='constrained')
    vmax = np.max(np.abs(images))
    for i, ax in enumerate(axs.flat):
        if i < len(images):
            im = ax.imshow(images[i], cmap=cmap, vmin=-vmax, vmax=vmax)
            ax.set_title(f'Component {i}')
        else:
            ax.axis('off')
        ax.spines[['bottom', 'right']].set_visible(True)
        ax.spines[['bottom', 'right']].set_color('gray')
        ax.spines[['top', 'left']].set_visible(False)
        ax.grid(True, color='gray')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(axis='both', bottom=False, left=False, )
    fig.colorbar(im, ax=axs, orientation='vertical', fraction=0.03, shrink=0.8,
        pad=0.01)
    plt.suptitle(title) if kwargs.get('title') else None
    plt.show()

def show_one_image(image: np.ndarray, 
    cmap='gray', colorbar=False):
    '''
    show one image
    '''
    figsize = (5, 5)
    fontsize = np.min(figsize) * 3
    plt.rc('xtick', labelsize=fontsize)
    plt.rc('ytick', labelsize=fontsize)
    fig, ax = plt.subplots()
    ax.imshow(image, cmap=cmap)
    ax.grid()
    if colorbar:
        plt.colorbar(ax.imshow(image, cmap=cmap), shrink=0.8)
    plt.show()