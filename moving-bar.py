#%%
import os
from os.path import join as pjoin
from tifffile import imread, imwrite
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter,median_filter
from NatMovie_utils import *
from wfield import *
from cor470 import enhance_df_f

#%% 定义一些路径
path_wfield = r'Y:\WF_VC_liuzhaoxi\24.06.20_H78\moving-bar\process\20240620-172655-wfield'
experiment = os.path.basename(path_wfield)[:15]
rawPath = pjoin(path_wfield,'..\\..\\raw')
path_out = pjoin(path_wfield, '..', experiment + '-mvbar')
os.makedirs(path_out, exist_ok=True)

direction_list=['right', 'down', 'left', 'up']
stim_len = 48 # bar=48 dots=138

#%% 根据trigger提取trials开始和结束帧
if not os.path.exists(pjoin(path_wfield, "trials.csv")):
    n_frame = min(len(os.listdir(pjoin(rawPath, experiment + "-405"))),
                  len(os.listdir(pjoin(rawPath, experiment + "-470"))))
    stimfile = pd.read_csv(pjoin(rawPath, experiment + ".csv"), header=None).values
    stim_delay = pd.read_csv(pjoin(rawPath, experiment + "-470Timestamp.csv"), header=None).values
    stim_delay = int(stim_delay[0] / 10)
    stim = np.zeros(n_frame)
    for i in range(n_frame):
        stim[i] = stimfile[(i * 10 + stim_delay), 0]

    offset = np.where(np.diff(stim) == 1)[0] + 1
    onset = np.where(np.diff(stim) == -1)[0] + 1
    offset = offset[1:]
    offset = np.append(offset, onset[-1] + stim_len)
    trials_csv = np.stack((np.arange(len(onset)), onset, offset, offset - onset), axis=1)
    np.savetxt(pjoin(path_wfield, "trials.csv"), trials_csv, delimiter=",")
else:
    print("trials.csv of "+experiment+" has existed.")

#%% 导入预处理后的tif数据和rf68刺激seq
U = np.load(pjoin(path_wfield, 'U.npy')).astype('float32')
SVTcorr = np.load(pjoin(path_wfield, 'SVTcorr.npy')).astype('float32')
frames_average = np.load(pjoin(path_wfield, 'frames_average.npy')).astype('float32')
trialfile = pd.read_csv(pjoin(path_wfield, 'trials.csv'), header=None).values.astype(int)
seq = pd.read_csv(r'D:\Zhaoxi\mouse_vision\code\WF\d4sequence.txt', header=None).values


#%%
def sorting_4direct(SVT, trigger, seq, stim_len):
    """
    data: ndarray,
    trigger: the index of each trigger
    seq: ndarray
    """
    nSVD = SVT.shape[0]
    n_stim = np.unique(seq).size
    n_trigger = trigger.size
    n_rep = int(n_trigger / n_stim)
    if n_trigger != n_stim * n_rep:
        print('n_trigger != n_stim * n_rep')
        return
    SVT_sorted = np.zeros((nSVD, stim_len, n_stim, n_rep))
    for i_trigger in range(n_trigger):
        i_rep = int(i_trigger / n_stim)
        SVT_sorted[:nSVD, :stim_len, seq[i_trigger]//90, i_rep] = SVT[:, trigger[i_trigger]:trigger[i_trigger] + stim_len]

    return SVT_sorted.astype('float16')


#%% sorting
# SVTcorr_sort维度：[nSVD, stim_len, n_stim, n_rep]
SVTcorr_sort = sorting_4direct(SVTcorr, trialfile[:, 1], seq[:, 0], stim_len).astype('float32')
n_stim = SVTcorr_sort.shape[2]
n_rep = SVTcorr_sort.shape[3]
width, height = U.shape[0:2]

#%% 导出各种tif
# tif_sort维度：[width, height, stim_len, n_stim, n_rep]
tif_sort = np.tensordot(U, SVTcorr_sort, axes=(2, 0)).astype('float32')
tif_mean = np.mean(tif_sort, axis=-1)
tif_with_mean = np.concatenate((tif_mean[..., np.newaxis], tif_sort), axis=-1)
tif_rep_reshape = np.concatenate(np.split(tif_with_mean, tif_with_mean.shape[-1], axis=-1),
                                 axis=1).squeeze().transpose(2, 0, 1, 3)
for i_dir in range(n_stim):
    print('start processing {}-{}'.format(i_dir + 1, direction_list[i_dir]))
    imwrite(pjoin(path_out, direction_list[i_dir] + '-rep-reshape.tif'),
            tif_rep_reshape[:, :, :, i_dir].astype('float32'), imagej=True)
print('export all rep-reshape-tifs')

for i_dir in range(n_stim):
    print('start processing {}-{}'.format(i_dir + 1, direction_list[i_dir]))
    imwrite(pjoin(path_out, direction_list[i_dir] + '-avg.tif'),
            tif_mean[:, :, :, i_dir].transpose(2, 0, 1).astype('float32'), imagej=True)
print('export all avg-tifs')

for i_dir in range(n_stim):
    tif = pjoin(path_out, direction_list[i_dir] + '-avg.tif')
    print('start '+direction_list[i_dir])
    img = imread(tif)
    img_enhance = enhance_df_f(img, frames_average)
    imwrite(tif[:-4]+'-enhance.tif', img_enhance.astype('uint16'), imagej=True)
    print('finish'+tif[:-4]+'-enhance.tif')


#%% 计算snr
snr = cal_snr(tif_sort, axis1=4, axis2=2)
subplot_movie_heatmap(snr, 2, 2, direction_list, path_outfile=pjoin(path_out,'moving-bar snr'), title='moving-bar snr', vmin=None, vmax=None,
                      cmap='hot', pixel_um=26)
# snr_filter = median_filter(snr,2)
# subplot_movie_heatmap(snr_filter, 2, 2, direction_list, path_outfile=pjoin(path_out,'moving-bar filter snr'), title='moving-bar filter snr', vmin=None, vmax=None,
#                       cmap='hot', pixel_um=26)

# %% prepare allen ccf map
from wfield import *

lmarks = load_allen_landmarks(pjoin(path_wfield, 'dorsal_cortex_landmarks.json'))
ccf_regions_reference, proj, brain_outline = allen_load_reference('dorsal_cortex')
# this loads the untransformed atlas
atlas_im, areanames, brain_mask = atlas_from_landmarks_file(pjoin(path_wfield, 'dorsal_cortex_landmarks.json'),
                                                            do_transform=True)
# this converts the reference to image space (unwarped)
ccf_regions_im = allen_transform_regions(lmarks['transform'], ccf_regions_reference,
                                         resolution=lmarks['resolution'],
                                         bregma_offset=lmarks['bregma_offset'])


# %% plot ccf map on average frame
frames_ave = np.load(pjoin(path_wfield, 'frames_average.npy'))[0]
merge_frame_size = (512, 512)  # (width, height)
fig = plt.figure(figsize=(merge_frame_size[0] / 128, merge_frame_size[1] / 128), dpi=128)
plt.imshow(frames_ave, cmap='gray')
for i, r in ccf_regions_im.iterrows():
    plt.plot(r['left_x'], r['left_y'], 'r', lw=0.2)
    plt.plot(r['right_x'], r['right_y'], 'r', lw=0.2)
    plt.text(r.left_center[0], r.left_center[1], r.acronym, color='w', va='center', fontsize=3, alpha=0.5, ha='center')

plt.axis('off')
fig.set_facecolor('white')
plt.savefig(pjoin(path_out, 'ccf.png'), bbox_inches='tight', pad_inches=0)
plt.show()

#%%
path_ccf_stim = pjoin(path_out, experiment + '-avg-ccf-stim')
os.makedirs(path_ccf_stim, exist_ok=True)

for direction in ['up', 'down', 'left', 'right']:
    stim_file = pjoin(r'D:\Zhaoxi\mouse_vision\code\WF\others', 'moving-bar-' + direction + '.mp4')
    tif_avg = pjoin(path_out, direction + '-avg' + '.tif')
    avg_ccf_stim_file = pjoin(path_ccf_stim, os.path.basename(tif_avg)[:-4] + '-ccf-stim.mp4')
    merge_ccf_stim(avg_ccf_stim_file, tif_avg, ccf_regions_im, stim_file=stim_file, tif_fps=10, trial_rep=1,
                   vmin=-0.04, vmax=0.04, text=direction + '-avg ')
print('\nfinish all avg merging')


for direction in ['up', 'down', 'left', 'right']:
    stim_file = pjoin(r'D:\Zhaoxi\mouse_vision\code\WF\others', 'moving-bar-' + direction + '.mp4')
    tif_avg = pjoin(path_out, 'tif', direction + '-avg-enhance' + '.tif')
    avg_ccf_stim_file = pjoin(path_ccf_stim, os.path.basename(tif_avg)[:-4] + '-ccf-stim.mp4')
    merge_ccf_stim(avg_ccf_stim_file, tif_avg, ccf_regions_im, stim_file=stim_file, tif_fps=10, trial_rep=1,
                   vmin=200, vmax=25000, text=direction + '-avg ')
print('\nfinish all avg merging')

