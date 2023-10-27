import numpy as np
import scipy
import matplotlib.pyplot as plt
from os.path import join as pjoin

#%%
def sorting(SVT, trigger, seq, stim_len):
    """
    data: ndarray,
    trigger: the index of each trigger
    seq: ndarray
    """
    nSVD = SVT.shape[0]
    n_seq = seq.size
    n_stim = np.unique(seq).size
    n_rep = int(n_seq / n_stim)
    n_trigger = trigger.size
    if n_trigger != n_stim * n_rep:
        print('n_trigger != n_stim * n_rep')
    SVT_sorted = np.zeros((nSVD, stim_len, n_stim, n_rep))
    for i_trigger in range(n_trigger):
        i_rep = int(i_trigger / n_stim)
        SVT_sorted[:nSVD, :stim_len, seq[i_trigger], i_rep] = SVT[:, trigger[i_trigger]:trigger[i_trigger] + stim_len]

    return SVT_sorted

#%%
def cal_snr(x,axis1,axis2):
    """
    axis1:重复，axis2:时间。
    """
    snr = np.nanvar(np.nanmean(x,axis=axis1),axis=axis2)/np.nanmean(np.nanvar(x,axis=axis1),axis=axis2)
    return snr

#%%
def find_peak(data, win_index=None, baseline=0, signal_deflect=1, axis=0):
    """
    axis参数指定时序所在维度。
    指定win_index时时序必须在第0维
    """
    if win_index is not None:
        win_index = [0,data.shape[axis]]
        if signal_deflect == 1:
            peak = np.max(data[win_index[0]:win_index[1]], axis=0) - baseline
        else:
            peak = baseline - np.min(data[win_index[0]:win_index[1]], axis=0)
    else:
        if signal_deflect == 1:
            peak = np.max(data,axis=axis) - baseline
        else:
            peak = baseline - np.min(data,axis=axis)

    peak = peak * (peak > 0)
    return peak

#%%
def idx_to_visual(rf_idx, para, n_row, n_col):
    import math
    row_indx = rf_idx[0]
    col_idx = rf_idx[1]
    x_offset = (col_idx + 0.5 - n_col / 2) * para['screen_width'] / n_col
    y_offset = -(row_indx + 0.5 - n_row / 2) * para['screen_height'] / n_row
    rf_pos_deg = [math.degrees(math.atan(x_offset / para['distance']))+para['screen_center'][0],
                  math.degrees(math.atan(y_offset / para['distance']))+para['screen_center'][1]]
    return rf_pos_deg

#%%
def analysis_rf(peak_amp, para, plot_results=False, plot_fitting=False):
    rf_row = 6
    rf_col = 8
    peak_amp = peak_amp
    img_width, img_height = peak_amp.shape[0:2]

    # separate on and off subfields
    n_rf_stim = rf_row * rf_col
    rf_on = peak_amp[:, :, 0:n_rf_stim].reshape((img_width, img_height, rf_col, rf_row)).transpose(3, 2, 0, 1)  # for on stimulus
    rf_off = peak_amp[:, :, n_rf_stim:n_rf_stim * 2].reshape((img_width, img_height, rf_col, rf_row)).transpose(3, 2, 0, 1)  # for off stimulus
    # results_rf['rf_on'] = rf_on
    # results_rf['rf_off'] = rf_off
    # rf_csi = (rf_on_max - rf_off_max) / (rf_on_max + rf_off_max)
    # results_rf['rf_csi'] = rf_csi
    '''
    if plot_results:
        folder_path = os.path.dirname(results_rf['file_path'])
        plot_rfs(results_rf['rf_on'], n_cols=10)
        save_fig(folder_path + '/rf_on')
        plot_rfs(results_rf['rf_off'], n_cols=10)
        save_fig(folder_path + '/rf_off')
    
    stim_center = results_rf['stim_center']    
    interp_num = 5
    rf_fit_list = []
    if 'rf_area_fit' not in results_rf:
        rf_pos_deg = np.zeros((2, n_rois))
        rf_area = np.zeros(n_rois)
        for i in range(n_rois):
            print(i)
            if rf_csi[i] > 0:
                rf_fit_para, rf_2d_fit = rf_process(rf_on[:, :, i], visual_center=stim_center, interp_num=interp_num,
                                                    plot=plot_fitting)
            else:
                rf_fit_para, rf_2d_fit = rf_process(rf_off[:, :, i], visual_center=stim_center, interp_num=interp_num,
                                                    plot=plot_fitting)
            rf_fit_list.append(rf_2d_fit)
            rf_pos_deg[0, i] = rf_fit_para['x0']
            rf_pos_deg[1, i] = rf_fit_para['y0']
            rf_area[i] = rf_fit_para['area']
        results_rf['rf_pos_deg_fit'] = rf_pos_deg
        results_rf['rf_area_fit'] = rf_area
        results_rf['rf_2d_fit'] = np.asarray(rf_fit_list)
    '''
    # if 'rf_pos_deg' not in results_rf:
    rf_pos_deg = np.zeros((2, img_width, img_height))
    rf_pos_row_col = np.zeros((2, img_width, img_height))
    for w in range(img_width):
        for h in range(img_height):
            rf_2d = np.maximum(rf_on[:, :, w, h], rf_off[:, :, w, h])
            rf_2d = rf_2d - np.mean(rf_2d)
            rf_2d = np.where(rf_2d < (np.max(rf_2d) / 3), 0, rf_2d)
            rf_pos_row_col[:, w, h] = scipy.ndimage.center_of_mass(rf_2d)
            rf_pos_deg[:, w, h] = idx_to_visual(rf_pos_row_col[:, w, h], para, rf_row, rf_col)

        # results_rf['rf_pos_deg'] = rf_pos_deg

    # return results_rf
    return rf_pos_deg, rf_pos_row_col

#%%
def amp_plot(amp, cmap='hsv', title=None, path=None, pixel_um=None, range=None):
    fig = plt.figure(figsize=[5, 5])
    if range is not None:
        plt.imshow(amp, cmap=cmap, interpolation='nearest',vmin=range[0], vmax=range[1])
    else:
        plt.imshow(amp, cmap=cmap, interpolation='nearest')
    plt.axis('off')
    plt.colorbar()
    plt.suptitle(title)
    fig.set_facecolor('white')
    if pixel_um is not None:
        # Calculate the length of the scale line in pixels
        scale_line_length = 1e3 / pixel_um
        line = plt.Line2D([0, 0 + scale_line_length], [amp.shape[1]-2, amp.shape[1]-2], color='black', linewidth=2)
        plt.gca().add_line(line)
        plt.text(scale_line_length / 2, amp.shape[1]+15, '1mm', color='black', fontsize=12, ha='center')
    if path is not None:
        plt.savefig(pjoin(path, title + '.png'))
    plt.show()

#%%
def plot_response_RF(data, window=[0, 20], title=None, path=None):
    n_rows = 6
    n_cols = 8
    ylim_max = data.max()
    ylim_min = data.min()
    temp = find_peak(data=np.mean(data, axis=2))
    vmin = np.min(temp)
    vmax = np.max(temp)
    data_on = data[:, 0:n_rows * n_cols, :]
    data_off = data[:, n_rows * n_cols:n_rows * n_cols * 2, :]

    fig = plt.figure(layout='constrained', figsize=(10, 10))
    subfigs = fig.subfigures(2, 2, wspace=0.07)

    # for plot the RF_on
    axis_trace = subfigs[0, 0].subplots(n_rows, n_cols)
    for col in range(n_cols):
        for row in range(n_rows):
            location_index = row + col * n_rows
            mean = np.mean(data_on[:, location_index, :], axis=1)
            y1 = mean + np.std(data_on[:, location_index, :], axis=1)
            y2 = mean - np.std(data_on[:, location_index, :], axis=1)
            axis_trace[row][col].fill_between(np.arange(data.shape[0]), y1, y2, alpha=0.3)
            axis_trace[row][col].plot(mean)
            axis_trace[row][col].spines['top'].set_visible(False)
            axis_trace[row][col].spines['right'].set_visible(False)
            axis_trace[row][col].spines['left'].set_visible(False)
            axis_trace[row][col].spines['bottom'].set_visible(False)
            axis_trace[row][col].set_ylim(ylim_min, ylim_max)
            axis_trace[row][col].set_xticks([])
            axis_trace[row][col].set_yticks([])
            axis_trace[row][col].hlines(y=0, xmin=window[0], xmax=window[1], color='r', linewidth=1, linestyle='--')

    axis_right = subfigs[0, 1].subplots()
    RF_on = np.zeros((n_rows, n_cols))
    temp = find_peak(data=np.mean(data_on, axis=2))
    RF_on[:, :] = temp.reshape((n_cols, n_rows)).T  # for on stimulus
    axis_right.imshow(RF_on, vmin=vmin, vmax=vmax)
    axis_right.set_title('RF_on')
    axis_right.set_xticks([])
    axis_right.set_yticks([])

    # for plot the RF_off
    axis_trace = subfigs[1, 0].subplots(n_rows, n_cols)
    for col in range(n_cols):
        for row in range(n_rows):
            location_index = row + col * n_rows
            mean = np.mean(data_off[:, location_index, :], axis=1)
            y1 = mean + np.std(data_off[:, location_index, :], axis=1)
            y2 = mean - np.std(data_off[:, location_index, :], axis=1)
            axis_trace[row][col].fill_between(np.arange(data.shape[0]), y1, y2, alpha=0.3)
            axis_trace[row][col].plot(mean)
            axis_trace[row][col].spines['top'].set_visible(False)
            axis_trace[row][col].spines['right'].set_visible(False)
            axis_trace[row][col].spines['left'].set_visible(False)
            axis_trace[row][col].spines['bottom'].set_visible(False)
            axis_trace[row][col].set_ylim(ylim_min, ylim_max)
            axis_trace[row][col].set_xticks([])
            axis_trace[row][col].set_yticks([])
            axis_trace[row][col].hlines(y=0, xmin=window[0], xmax=window[1], color='r', linewidth=1, linestyle='--')

    axis_right = subfigs[1, 1].subplots()
    RF_off = np.zeros((n_rows, n_cols))
    temp = find_peak(data=np.mean(data_off, axis=2))
    RF_off[:, :] = temp.reshape((n_cols, n_rows)).T  # for on stimulus
    axis_right.imshow(RF_off, vmin=vmin, vmax=vmax)
    axis_right.set_title('RF_off')
    axis_right.set_xticks([])
    axis_right.set_yticks([])

    plt.suptitle(title)
    fig.set_facecolor('white')
    if path is not None:
        plt.savefig(pjoin(path,title+'.png'))
    plt.show()
