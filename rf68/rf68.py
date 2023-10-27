import os
from os.path import join as pjoin
from tifffile import imread, imwrite
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rf68_utils import *
from wfield import *


#%%
para = {'distance':150, #mm
        'visual_center': (45,20), # (az,el)deg
        'screen_center':(45,20), # (az,el)deg
        'screen_width':595, #mm
        'screen_height':336, #mm
        # 'screen_width_pix':1920, # in pixel
        # 'screen_height_pix':1080, # in pixel
        'sampling_rate': 1/0.1,
        'micrometer_per_pixel': 26,
        'imag_size_pixel':np.array([256,256]), # in pixel
        }
para['imag_size'] = para['imag_size_pixel']*para['micrometer_per_pixel'] # in micrometer

# mergePath = 'Z:\\WF_VC_liuzhaoxi\\23.10.20_G360_gp4.3\\rf68\\process\\20231020-204235-merged'
mergePath = 'D:\\Zhaoxi\\mouse_vision\\data\\20231020-204235-merged'


'''
rawPath = 'Z:\\WF_VC_liuzhaoxi\\23.10.20_G360_gp4.3\\rf68\\raw'
experiment = '20231020-204235'

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
offset = np.append(offset, onset[-1]+20)
trials_csv = np.stack((np.arange(len(onset)), onset, offset, offset - onset), axis=1)
np.savetxt(pjoin(mergePath, "trials.csv"), trials_csv, delimiter=",")
'''
#%%
U = np.load(pjoin(mergePath, 'U.npy'))
SVTcorr = np.load(pjoin(mergePath, 'SVTcorr.npy'))
SVTcorr[:,11403:11403+5] = np.nan
frames_average = np.load(pjoin(mergePath, 'frames_average.npy'))
trialfile = pd.read_csv(pjoin(mergePath, 'trials.csv'), header=None).values.astype(int)
seq = pd.read_csv('seq6x8_bk.txt', header=None).values
is_off = seq[:, 1] == 1
seq[:, 0] = np.where(is_off, seq[:, 0] + 48, seq[:, 0])

stim_len = 20

#%%
# SVTcorr_sort维度：[nSVD, stim_len, n_stim, n_rep]
SVTcorr_sort = sorting(SVTcorr, trialfile[:, 1], seq[:, 0], stim_len)
# tif_sort维度：[width, height, stim_len, n_stim, n_rep]
tif_sort = np.tensordot(U, SVTcorr_sort, axes=(2, 0)).astype('float32')
# tif_sort=tif_sort[75:180,75:195,:,:,:]
# imwrite(pjoin(mergePath,'loc1-ave.tif'),tif_sort[:,:,:,0,-1].transpose(2,0,1).astype('float32'), imagej=True)
n_stim = tif_sort.shape[3]
n_rep = tif_sort.shape[4]
width, height = tif_sort.shape[0:2]

peak_amp = find_peak(np.nanmean(tif_sort, axis=-1), axis=2)
snr = cal_snr(tif_sort, axis1=4, axis2=2)
pixel_snr=np.max(snr,axis=2)
pixel_peak=np.max(peak_amp, axis=2)
# imwrite('pixel_snr.tif',pixel_snr.astype('float32'), imagej=True)
# imwrite('stimuli_peak.tif',peak_amp.transpose(2,0,1).astype('float32'), imagej=True)
# imwrite('pixel_peak.tif',pixel_peak.astype('float32'), imagej=True)

#%%
rf_pos_deg, rf_pos_row_col = analysis_rf(peak_amp, para)
amp_plot(rf_pos_deg[0, :], cmap='hsv', title='rf68_azimuth', path='.', pixel_um=para['micrometer_per_pixel'])
amp_plot(rf_pos_deg[1, :], title='rf68_elevation', path='.', pixel_um=para['micrometer_per_pixel'])
amp_plot(pixel_snr, cmap='hot', title='rf68_snr', path='.', pixel_um=para['micrometer_per_pixel'])

#%%
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
plt.savefig('az_el_unfiltered.png')
plt.show()
#%%
from scipy.ndimage import median_filter
fig = plt.figure()
plt.imshow(median_filter(visual_sign_map(rf_pos_deg[0, :], rf_pos_deg[1, :]), 20),
           cmap='RdBu_r', clim=[-1, 1])
plt.colorbar(shrink=0.5)
plt.axis('off')
plt.suptitle('rf68_signmap')
fig.set_facecolor('white')
plt.savefig('phasemap.png')
plt.show()
#%%
'''
max_indices = np.argpartition(pixel_snr, -10, axis=None)[-10:]
max_indices = np.unravel_index(max_indices, pixel_snr.shape)
max_indices=np.asarray(max_indices)
'''
# 计算最后一维的平均值，忽略NaN值
def NaN2mean(data, axis=-1):
    mean_values = np.nanmean(data, axis=axis, keepdims=True)
    data = np.where(np.isnan(data), mean_values, data)
    return data

plot_response_RF(NaN2mean(tif_sort[99,191,:,:,:]), window=[0, 20], title='traces_and_RF', path='.')
'''
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
plt.scatter(99, 191, s=100, facecolors='none', edgecolors='black', marker='o')
plt.savefig('plot_selected_pixel.png', bbox_inches='tight')
plt.show()
'''
#%%
print('end')
