#%%
import NeuroAnalysisTools
import NeuroAnalysisTools.core.FileTools as ft
import NeuroAnalysisTools.RetinotopicMapping as rm
import pickle
import os
import numpy as np
import pandas as pd
from tifffile import imread, imwrite
import matplotlib.pyplot as plt
import cv2
from glob import glob
from NatMovie_utils import *

#%% set path and parameter
path_wfield = r'/home/lzx/data/WF_VC_liuzhaoxi/24.07.08_Q20/retinotopy/process/20240708-192659-wfield'

experiment = os.path.basename(path_wfield)[:15]
rawPath = pjoin(path_wfield, '..\\..\\raw')
path_out = pjoin(path_wfield, '..', experiment + '-retinotopy')
os.makedirs(path_out, exist_ok=True)
path_retinotopy = path_wfield[:-6] + 'retinotopy'

# %% load patch data

with open(pjoin(path_retinotopy, 'retinotopy_out.pkl'), 'rb') as f:
    retino = pickle.load(f)
f.close()
n_patch = len(retino['finalPatchesMarked'])

# %% merge tif & patch & stim

path_patch_stim = pjoin(path_out, experiment + '-patch-stim')
os.makedirs(path_patch_stim, exist_ok=True)

for direction in ['up', 'down', 'left', 'right']:
    stim_file = pjoin(r'D:\Zhaoxi\mouse_vision\code\retinotopy', 'stim-' + direction + '.mp4')
    tif_avg = pjoin(path_out, 'avg_' + direction + '.tif')
    avg_patch_stim_file = pjoin(path_patch_stim, os.path.basename(tif_avg)[:-4] + '-patch-stim.mp4')
    merge_patch_stim(avg_patch_stim_file, tif_avg, stim_file, clip=0.02, patches=retino['finalPatchesMarked'], ncol=1,
                     text=direction + '-avg ')
    print('finish merging avg-' + direction)
print('\nfinish all avg merging')

for direction in ['up', 'down', 'left', 'right']:
    stim_file = pjoin(r'D:\Zhaoxi\mouse_vision\code\retinotopy', 'stim-' + direction + '.mp4')
    tif_rep = pjoin(path_out, 'rep_' + direction + '.tif')
    rep_patch_stim_file = pjoin(path_patch_stim, os.path.basename(tif_rep)[:-4] + '-patch-stim.mp4')
    merge_patch_stim(rep_patch_stim_file, tif_rep, stim_file, clip=0.05, patches=retino['finalPatchesMarked'], ncol=1,
                     trial_rep=10, text=direction + '-rep ')
    print('finish merging rep-' + direction)
print('\nfinish all rep merging')

# %% prepare allen ccf map
from wfield import *

lmarks = load_allen_landmarks(pjoin(path_wfield, 'dorsal_cortex_landmarks.json'))
ccf_regions_reference, proj, brain_outline = allen_load_reference('dorsal_cortex')
# this loads the untransformed atlas
atlas_im, areanames, brain_mask = atlas_from_landmarks_file(pjoin(path_wfield, 'dorsal_cortex_landmarks.json'),do_transform = True)
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
    plt.plot(r['right_x'],r['right_y'],'r',lw=0.2)
    plt.text(r.left_center[0], r.left_center[1],
             r.acronym, color='w', va='center',
             fontsize=6, alpha=0.5, ha='center')
plt.axis('off')
fig.set_facecolor('white')
plt.savefig(pjoin(path_retinotopy, 'ccf.png'), bbox_inches='tight', pad_inches=0)
plt.show()

# %% merge tif & ccf & stim

path_ccf_stim = pjoin(path_out, experiment + '-ccf-stim')
os.makedirs(path_ccf_stim, exist_ok=True)

for direction in ['up', 'down', 'left', 'right']:
    stim_file = pjoin(r'/home/lzx/data/WF_VC_liuzhaoxi/code/WF/others', 'stim-' + direction + '.mp4')
    tif_avg = pjoin(path_out, 'tif', 'avg_' + direction + '.tif')
    avg_ccf_stim_file = pjoin(path_ccf_stim, os.path.basename(tif_avg)[:-4] + '-ccf-stim.mp4')
    merge_ccf_stim(avg_ccf_stim_file, tif_avg, ccf_regions_im, stim_file=stim_file, tif_fps=10, trial_rep=1,
                   vmin=-0.05, vmax=0.05, text=direction + '-avg ')
print('\nfinish all avg merging')

for direction in ['up', 'down', 'left', 'right']:
    stim_file = pjoin(r'D:\Zhaoxi\mouse_vision\code\retinotopy', 'stim-' + direction + '.mp4')
    tif_rep = pjoin(path_out, 'tif', 'rep_' + direction + '.tif')
    rep_ccf_stim_file = pjoin(path_ccf_stim, os.path.basename(tif_rep)[:-4] + '-ccf-stim.mp4')
    merge_ccf_stim(rep_ccf_stim_file, tif_rep, ccf_regions_im, stim_file=stim_file, tif_fps=10, trial_rep=10,
                   vmin=-0.1, vmax=0.1, text=direction + '-rep ')
print('\nfinish all rep merging')

for direction in ['up', 'down', 'left', 'right']:
    stim_file = pjoin(r'D:\Zhaoxi\mouse_vision\code\retinotopy', 'stim-' + direction + '.mp4')
    tif_rep = pjoin(path_out, 'tif', 'rep_' + direction + '_reshape.tif')
    rep_ccf_stim_file = pjoin(path_ccf_stim, os.path.basename(tif_rep)[:-4] + '-ccf-stim.mp4')
    merge_ccf_stim(rep_ccf_stim_file, tif_rep, ccf_regions_im, stim_file=stim_file, ncol=5, nrow=2, tif_fps=10,
                   vmin=-0.1, vmax=0.1, trial_rep=1, text=direction + '-rep-reshape ')
print('\nfinish all rep reshape merging')

# %% merge enhance-tif & ccf & stim

path_ccf_stim = pjoin(path_out, experiment + '-enhance-ccf-stim')
os.makedirs(path_ccf_stim, exist_ok=True)

for direction in ['up', 'down', 'left', 'right']:
    stim_file = pjoin(r'D:\Zhaoxi\mouse_vision\code\retinotopy', 'stim-' + direction + '.mp4')
    tif_avg = pjoin(path_out, 'tif-enhance', 'avg_' + direction + '_enhance.tif')
    avg_ccf_stim_file = pjoin(path_ccf_stim, os.path.basename(tif_avg)[:-4] + '-ccf-stim.mp4')
    merge_ccf_stim(avg_ccf_stim_file, tif_avg, ccf_regions_im, stim_file=stim_file, tif_fps=10,
                   vmin=200, vmax=25000, trial_rep=1, text=direction + '-avg ')
print('\nfinish all avg merging')

for direction in ['up', 'down', 'left', 'right']:
    stim_file = pjoin(r'D:\Zhaoxi\mouse_vision\code\retinotopy', 'stim-' + direction + '.mp4')
    tif_rep = pjoin(path_out, 'tif-enhance', 'rep_' + direction + '_enhance.tif')
    rep_ccf_stim_file = pjoin(path_ccf_stim, os.path.basename(tif_rep)[:-4] + '-ccf-stim.mp4')
    merge_ccf_stim(rep_ccf_stim_file, tif_rep, ccf_regions_im, stim_file=stim_file, tif_fps=10,
                   vmin=200, vmax=25000, trial_rep=10, text=direction + '-rep ')
print('\nfinish all rep merging')

for direction in ['up', 'down', 'left', 'right']:
    stim_file = pjoin(r'D:\Zhaoxi\mouse_vision\code\retinotopy', 'stim-' + direction + '.mp4')
    tif_rep = pjoin(path_out, 'tif-enhance', 'rep_' + direction + '_enhance_reshape.tif')
    rep_ccf_stim_file = pjoin(path_ccf_stim, os.path.basename(tif_rep)[:-4] + '-ccf-stim.mp4')
    merge_ccf_stim(rep_ccf_stim_file, tif_rep, ccf_regions_im, stim_file=stim_file, ncol=5, nrow=2, tif_fps=10,
                   vmin=200, vmax=25000, trial_rep=1, text=direction + '-rep-reshape ')
print('\nfinish all rep reshape merging')

