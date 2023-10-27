import numpy as np
import scipy
import matplotlib.pyplot as plt
from os.path import join as pjoin


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


def cal_snr(x,axis1,axis2):
    """
    axis1:重复，axis2:时间。
    """
    snr = np.nanvar(np.nanmean(x,axis=axis1),axis=axis2)/np.nanmean(np.nanvar(x,axis=axis1),axis=axis2)
    return snr


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


def idx_to_visual(rf_idx, para):
    x = rf_idx[0]
    y = rf_idx[1]
    visual_origin = np.array([para['stim_center'][0] - para['n_cols'] / 2 * para['size'],
                              para['stim_center'][1] + para['n_rows'] / 2 * para['size']])
    rf_pos_deg = np.zeros(2)
    rf_pos_deg[0] = visual_origin[0] + (y + 0.5) * para['size']
    rf_pos_deg[1] = visual_origin[1] - (x + 0.5) * para['size']
    # x+=1
    # y+=1
    # rf_pos_deg = [(y-(para['n_cols']+1)/2)*para['size']+para['visual_center'][0],
    #                         -(x-(para['n_rows']+1)/2)*para['size']+para['visual_center'][1]
    return rf_pos_deg


def analysis_rf(peak_amp, plot_results=False, plot_fitting=False):
    # separate on and off subfields
    rf_row = 6
    rf_col = 8
    peak_amp = peak_amp
    width, height = peak_amp.shape[0:2]
    rf_on = np.zeros((width, height, rf_row, rf_col))
    rf_off = np.zeros((width, height, rf_row, rf_col))
    n_rf_stim = rf_row * rf_col
    rf_on = peak_amp[:, :, 0:n_rf_stim].reshape((width, height, rf_col, rf_row)).transpose(3, 2, 0, 1)  # for on stimulus
    rf_off = peak_amp[:, :, n_rf_stim:n_rf_stim * 2].reshape((width, height, rf_col, rf_row)).transpose(3, 2, 0, 1)  # for off stimulus
    # results_rf['rf_on'] = rf_on
    # results_rf['rf_off'] = rf_off
    rf_on_max = np.max(rf_on, axis=(0, 1))
    rf_off_max = np.max(rf_off, axis=(0, 1))
    # rf_csi = (rf_on_max - rf_off_max) / (rf_on_max + rf_off_max)
    rf_csi = rf_on_max - rf_off_max
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
    # if 'rf_pos_deg_cal' not in results_rf:
    rf_pos_deg_cal = np.zeros((2, width, height))
    for w in range(width):
        for h in range(height):
            if rf_csi[w, h] > 0:
                rf_2d = np.copy(rf_on[:, :, w, h])
            else:
                rf_2d = np.copy(rf_off[:, :, w, h])
            rf_2d = rf_2d - np.mean(rf_2d)
            rf_2d = np.where(rf_2d < (np.max(rf_2d) / 30), 0, rf_2d)
            # x, y = scipy.ndimage.center_of_mass(rf_2d)
            # rf_pos_deg_cal[:, i] = idx_to_visual([x, y], results_rf)
            rf_pos_deg_cal[:, w, h] = scipy.ndimage.center_of_mass(rf_2d)
        # results_rf['rf_pos_deg_cal'] = rf_pos_deg_cal

    # return results_rf
    return rf_pos_deg_cal


def amp_plot(amp, cmap='viridis', title=None, path=None):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlim(0, amp.shape[1])  # 根据数据的宽度设置 x 轴范围
    ax.set_ylim(0, amp.shape[0])  # 根据数据的高度设置 y 轴范围
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    s = ax.imshow(amp, cmap=cmap, interpolation='nearest')
    fig.set_size_inches(8, 8)
    ax.invert_yaxis()
    fig.colorbar(s)
    plt.suptitle(title)
    plt.savefig(pjoin(path, title+'.png'))
    plt.show()

