# trial_onsets.npy is a Nx3 array. The columns are: trial number, onset of the trial (frame index), offset of the trial

import numpy as np
import pandas as pd
import os
from os.path import join as pjoin

path = 'Z:\\WF_VC_liuzhaoxi\\23.9.4_G360_gp4.3\\3s-flash'
experiment = '20230904-205549'
n_frame = min(len(os.listdir(pjoin(path, 'raw', experiment + "-405"))), len(os.listdir(pjoin(path, 'raw', experiment + "-470"))))
mergePath = 'Z:\\WF_VC_liuzhaoxi\\23.9.4_G360_gp4.3\\3s-flash\\process\\20230904-205549-merged'

stimfile = pd.read_csv(pjoin(path, 'raw', experiment + ".csv"), header=None).values
stim = np.zeros(n_frame)
for i in range(n_frame):
    stim[i] = stimfile[(i * 20), 0]

onset = np.where(np.diff(stim) == 1)[0] + 1
offset = np.where(np.diff(stim) == -1)[0] + 1
nframes_pre = 20 # 预留的刺激前gap，用来计算baseline
trial_onsets = np.stack((np.arange(len(onset)), onset-nframes_pre, offset), axis=1)
trials_csv = np.stack((np.arange(len(onset)), onset, offset, offset-onset), axis=1)

np.save(pjoin('Z:\\WF_VC_liuzhaoxi\\test\\GP4.3\\trials', 'trial_onsets.npy'), trial_onsets)
np.savetxt(pjoin('Z:\\WF_VC_liuzhaoxi\\test\\GP4.3\\trials', "trials.csv"), trials_csv, delimiter=",")

print("generate trial_onsets.npy of "+experiment)

