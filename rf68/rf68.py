import os
from os.path import join as pjoin
from tifffile import imread, imwrite
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rf68_utils import *

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

U = np.load(pjoin(mergePath, 'U.npy'))
SVTcorr = np.load(pjoin(mergePath, 'SVTcorr.npy'))
SVTcorr[:,11403:11403+5] = np.nan
frames_average = np.load(pjoin(mergePath, 'frames_average.npy'))
trialfile = pd.read_csv(pjoin(mergePath, 'trials.csv'), header=None).values.astype(int)
seq = pd.read_csv('seq6x8_bk.txt', header=None).values
is_off = seq[:, 1] == 1
seq[:, 0] = np.where(is_off, seq[:, 0] + 48, seq[:, 0])

stim_len = 20


# SVTcorr_sort维度：[nSVD, stim_len, n_stim, n_rep]
SVTcorr_sort = sorting(SVTcorr, trialfile[:, 1], seq[:, 0], stim_len)
# tif_sort维度：[width, height, stim_len, n_stim, n_rep]
tif_sort = np.tensordot(U, SVTcorr_sort, axes=(2, 0))
# tif_sort=tif_sort[75:180,75:195,:,:,:]
# imwrite(pjoin(mergePath,'loc1-ave.tif'),tif_sort[:,:,:,0,-1].transpose(2,0,1).astype('float32'), imagej=True)
n_stim = tif_sort.shape[3]
n_rep = tif_sort.shape[4]
width, height = tif_sort.shape[0:2]

peak_amp = find_peak(np.nanmean(tif_sort, axis=-1), axis=2)
snr = cal_snr(tif_sort, axis1=4, axis2=2)
pixel_snr=np.max(snr,axis=2)
pixel_max=np.max(peak_amp,axis=2)
# imwrite('peak_amp.tif',peak_amp.transpose(2,0,1).astype('float32'), imagej=True)
# imwrite('pixel_max.tif',pixel_max.astype('float32'), imagej=True)

rf_pos_deg_cal = analysis_rf(peak_amp)
amp_plot(rf_pos_deg_cal[0,:], title='rf68_azimuth',path='.')
amp_plot(rf_pos_deg_cal[1,:], title='rf68_elevation',path='.')


print()
