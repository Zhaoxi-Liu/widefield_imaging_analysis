import os
from os.path import join as pjoin
from tifffile import imread, imwrite
import numpy as np
import pandas as pd
from wfield import *


rawPath = 'Z:\\WF_VC_liuzhaoxi\\23.10.20_G360_gp4.3\\rf68\\raw'
experiment = '20231020-204235'
mergePath = 'Z:\\WF_VC_liuzhaoxi\\23.10.20_G360_gp4.3\\rf68\\process\\20231020-204235-merged'
'''
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
print("generate trial_onsets.npy of " + experiment)
'''





U = np.load(pjoin(mergePath, 'U.npy'))
SVTcorr = np.load(pjoin(mergePath, 'SVTcorr.npy'))
SVTcorr[:,11403:11403+5] = np.nan

# frames_average = np.load(pjoin(path, 'frames_average.npy'))
# data = reconstruct(U, SVTcorr)

trialfile = pd.read_csv(pjoin(mergePath, 'trials.csv'), header=None).values.astype(int)
stim_len = 20

seq = pd.read_csv('seq6x8_bk.txt', header=None).values
is_off = seq[:, 1] == 1
seq[:, 0] = np.where(is_off, seq[:, 0] + 48, seq[:, 0])

def sorting(data, trigger, seq, stim_len):
    '''
    data: ndarray,
    trigger: the index of each trigger
    seq: ndarray
    '''
    nSVD = data.shape[0]
    n_seq = seq.size
    n_stim = np.unique(seq).size
    n_rep = int(n_seq / n_stim)
    n_trigger = trigger.size
    if n_trigger != n_stim * n_rep:
        print('n_trigger != n_stim * n_rep')
    data_sorted = np.zeros((nSVD, stim_len, n_stim, n_rep + 1))
    for i_trigger in range(n_trigger):
        i_rep = int(i_trigger / n_stim)
        data_sorted[:nSVD, :stim_len, seq[i_trigger], i_rep] = data[:, trigger[i_trigger]:trigger[i_trigger] + stim_len]

    data_sorted[:nSVD, :stim_len, :n_stim, n_rep] = np.mean(data_sorted[:, :, :, :n_rep], axis=3)

    return data_sorted

SVT_sort = sorting(SVTcorr, trialfile[:,1], seq[:,0], stim_len)

tif_sort = np.tensordot(U, SVT_sort, axes=([2], [0])).transpose(2,0,1,3,4)
imwrite(pjoin(mergePath,'loc1-ave.tif'),tif_sort[:,:,:,0,-1].astype('float32'), imagej=True)

print()
