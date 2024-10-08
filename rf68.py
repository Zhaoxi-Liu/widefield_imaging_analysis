# import sys
# sys.path.extend(['D:\\Zhaoxi\\mouse_vision\\code\\WF\\rf68'])

#%%
import os
from os.path import join as pjoin
from tifffile import imread, imwrite
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rf68_utils import *
from wfield import *
from scipy.ndimage import gaussian_filter,median_filter

#%%
para = {'distance':150, #mm
        'visual_center': (45,25), # (az,el)deg
        'screen_center':(45,25), # (az,el)deg
        'screen_width':595, #mm
        'screen_height':336, #mm
        # 'screen_width_pix':1920, # in pixel
        # 'screen_height_pix':1080, # in pixel
        'sampling_rate': 1/0.1,
        'micrometer_per_pixel': 26,
        'imag_size_pixel':np.array([256,256]), # in pixel
        }
para['imag_size'] = para['imag_size_pixel']*para['micrometer_per_pixel'] # in micrometer
x=math.degrees(math.atan( 1/2*para['screen_width'] / para['distance'] ))
y=math.degrees(math.atan( 1/2*para['screen_height'] / para['distance'] ))
para['az_range'] = (para['visual_center'][0]-x, para['visual_center'][0]+x)
para['el_range'] = (para['visual_center'][1]-y, para['visual_center'][1]+y)

stim_len = 20

#%% 定义一些路径
path_wfield = r'Y:\WF_VC_liuzhaoxi\23.11.16_C33\rf68\process\20231116-141022-wfield'
experiment = os.path.basename(path_wfield)[:15]
rawPath = pjoin(path_wfield,'..\\..\\raw')
path_out = pjoin(path_wfield, '..', experiment + '-rf68')
os.makedirs(path_out, exist_ok=True)

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
SVTcorr[:,1344:1344+5] = np.nan
frames_average = np.load(pjoin(path_wfield, 'frames_average.npy')).astype('float32')
trialfile = pd.read_csv(pjoin(path_wfield, 'trials.csv'), header=None).values.astype(int)
seq = pd.read_csv(r'D:\Zhaoxi\mouse_vision\code\WF\seq6x8.txt', header=None).values
is_off = seq[:, 1] == 1
seq[:, 0] = np.where(is_off, seq[:, 0] + 48, seq[:, 0])

#%% sorting
# SVTcorr_sort维度：[nSVD, stim_len, n_stim, n_rep]
SVTcorr_sort = sorting_rf68(SVTcorr, trialfile[:, 1], seq[:, 0], stim_len).astype('float32')
# tif_sort维度：[width, height, stim_len, n_stim, n_rep]
tif_sort = np.tensordot(U, SVTcorr_sort, axes=(2, 0)).astype('float32')
# imwrite(pjoin(path_wfield,'loc1-ave.tif'),tif_sort[:,:,:,0,-1].transpose(2,0,1).astype('float32'), imagej=True)
n_stim = SVTcorr_sort.shape[2]
n_rep = SVTcorr_sort.shape[3]
width, height = U.shape[0:2]

#%% 计算snr
snr = cal_snr(tif_sort, axis1=4, axis2=2)
pixel_snr=np.max(snr,axis=2)
# pixel_snr_filter = median_filter(pixel_snr,1)
amp_plot(pixel_snr, cmap='hot', title='rf68_snr', path_out=path_out, pixel_um=para['micrometer_per_pixel'])
imwrite(pjoin(path_out, 'rf68_snr.tif'),pixel_snr.astype('float32'), imagej=True)

#%% 计算peak，根据peak对应的刺激位置画出phasemap
peak_amp = find_peak(np.nanmean(tif_sort, axis=-1), axis=2)
pixel_peak = np.max(peak_amp, axis=2)
amp_plot(pixel_peak, cmap='hot', title='rf68_peak', path_out=path_out, pixel_um=para['micrometer_per_pixel'])
# imwrite('stimuli_peak.tif',peak_amp.transpose(2,0,1).astype('float32'), imagej=True)
# imwrite('pixel_peak.tif',pixel_peak.astype('float32'), imagej=True)
rf_pos_deg, rf_pos_row_col = analysis_rf(peak_amp, para)
amp_plot(rf_pos_deg[0, :], cmap='hsv', title='rf68_azimuth', path_out=path_out, pixel_um=para['micrometer_per_pixel'])
amp_plot(rf_pos_deg[1, :], title='rf68_elevation', path_out=path_out, pixel_um=para['micrometer_per_pixel'])

#%% 结合phasemap和pixel_snr画图
fig = plt.figure(figsize=[10, 5])
rf68_az = fig.add_subplot(1, 2, 1)
plt.imshow(im_fftphase_hsv([pixel_snr, rf_pos_row_col[1, :]]))
plt.axis('off')
rf68_az.set_title('rf68_az')
rf68_el = fig.add_subplot(1, 2, 2)
plt.imshow(im_fftphase_hsv([pixel_snr, rf_pos_row_col[0, :]]))
plt.axis('off')
rf68_el.set_title('rf68_el')
fig.set_facecolor('white')
plt.savefig(pjoin(path_out, 'phasemap_unfiltered.png'), bbox_inches='tight')
plt.show()

#%% 根据phasemap算signmap
from scipy.ndimage import median_filter
fig = plt.figure()
plt.imshow(median_filter(visual_sign_map(rf_pos_deg[0, :], rf_pos_deg[1, :]), 20),
           cmap='RdBu_r', clim=[-1, 1])
plt.colorbar(shrink=0.5)
plt.axis('off')
plt.suptitle('rf68_signmap')
fig.set_facecolor('white')
plt.savefig(pjoin(path_out, 'signmap.png'), bbox_inches='tight')
plt.show()

#%%
# 计算最后一维的平均值，忽略NaN值
def NaN2mean(data, axis=-1):
    mean_values = np.nanmean(data, axis=axis, keepdims=True)
    data = np.where(np.isnan(data), mean_values, data)
    return data

#%% 找出最大10个snr的pixel
max_indices = np.argpartition(pixel_snr, -10, axis=None)[-10:]
max_indices = np.asarray(np.unravel_index(max_indices, pixel_snr.shape))

#%% 圈出所选pixel
n_max = -1   #画排第几大的pixel
fig = plt.figure(figsize=[5, 5])
plt.imshow(pixel_snr, cmap='hot', interpolation='nearest')
plt.axis('off')
fig.set_facecolor('white')
pixel_um =26
# Calculate the length of the scale line in pixels
scale_line_length = 1e3 / pixel_um
line = plt.Line2D([0, 0 + scale_line_length], [pixel_snr.shape[1]-2, pixel_snr.shape[1]-2], color='black', linewidth=2)
plt.gca().add_line(line)
plt.text(scale_line_length / 2, pixel_snr.shape[1]+15, '1mm', color='black', fontsize=12, ha='center')
plt.scatter(max_indices[1, n_max], max_indices[0, n_max], s=50, facecolors='none', edgecolors='black', marker='o')
plt.savefig(pjoin(path_out, 'plot_selected_pixel.png'), bbox_inches='tight')
plt.show()

#%% 画RF
plot_response_RF(NaN2mean(tif_sort[max_indices[1, n_max], max_indices[0, n_max], :, :, :]), window=[0, 20], title='traces_and_RF', path_out=path_out)

#%%
print('end')

