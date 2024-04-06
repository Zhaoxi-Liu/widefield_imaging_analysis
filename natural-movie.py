# %%
import NeuroAnalysisTools
import NeuroAnalysisTools.core.FileTools as ft
import NeuroAnalysisTools.RetinotopicMapping as rm
import pickle
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap, Normalize
from NatMovie_utils import *
import os
from tifffile import imwrite
from scipy.ndimage import gaussian_filter, median_filter


# %% set path and parameter
path_wfield = r'Y:\WF_VC_liuzhaoxi\24.04.03_C92\natural-movie\process\20240403-161209-wfield'
path_retinotopy = r'Y:\WF_VC_liuzhaoxi\24.04.03_C92\retinotopy\process\20240403-170354-retinotopy'

experiment = os.path.basename(path_wfield)[:15]
rawPath = pjoin(path_wfield, '..\\..\\raw')
path_out = pjoin(path_wfield, '..', experiment + '-natural-movie')
os.makedirs(path_out, exist_ok=True)
movie_folder = pjoin(rawPath,'natural_movies')
movie_list = pd.read_csv(pjoin(movie_folder, 'movie_list.txt'), header=None).values
n_movie = movie_list.size

n_frame = 150  # 帧

#%% load data
U = np.load(pjoin(path_wfield, 'U.npy')).astype('float32')
SVTcorr = np.load(pjoin(path_wfield, 'SVTcorr.npy')).astype('float32')
frames_average = np.load(pjoin(path_wfield, 'frames_average.npy')).astype('float32')
trialfile = pd.read_csv(pjoin(path_wfield, 'trials.csv'), header=None).values.astype(int)

#%%
with open(pjoin(path_retinotopy, 'retinotopy_out.pkl'), 'rb') as f:
    retino = pickle.load(f)
f.close()

# %% sorting
# SVTcorr_sort维度：[nSVD, n_frame, n_movie, n_rep]
SVTcorr_sort = sorting_NatMov(SVTcorr, trialfile[:, 1], n_movie, n_frame).astype('float32')
print('SVTcorr_sort.shape: (nSVD, n_frame, n_movie, n_rep) ', SVTcorr_sort.shape)
n_rep = SVTcorr_sort.shape[-1]

# %% reconstruction and calculate SNR
# tif_sort维度：[width, height, n_frame, n_movie, n_rep]
# tif_sort = np.tensordot(U, SVTcorr_sort, axes=(2, 0)).astype('float32')
os.makedirs(pjoin(path_out, experiment+'-tif'), exist_ok=True)
snr = np.empty((U.shape[0], U.shape[1], n_movie))
for i in range(n_movie):
    print('start processing {}-{}'.format(i+1, str(movie_list[i])[2:-6]))
    tif_imovie = np.tensordot(U, SVTcorr_sort[:, :, i, :], axes=(2, 0)).astype('float32')
    snr[:, :, i] = cal_snr(tif_imovie, axis1=3, axis2=2)
    tif_imovie_rep_mean = np.mean(tif_imovie, axis=3)
    tif_imovie_with_mean = np.concatenate((tif_imovie, tif_imovie_rep_mean[..., np.newaxis]), axis=3)
    tif_rep = np.concatenate(np.split(tif_imovie_with_mean, tif_imovie_with_mean.shape[3], axis=3), axis=1).squeeze().transpose(2, 0, 1)
    imwrite(pjoin(path_out, experiment+'-tif', str(movie_list[i])[2:-6]+'.tif'), tif_rep, imagej=True)
print('export all rep-tifs')
np.save(pjoin(path_out,'snr.npy'),snr)

#%% plot snr
snr=np.load(pjoin(path_out,'snr.npy'))
plot_movie_snr(snr, 4, 4, movie_list, patches=retino['finalPatchesMarked'], vmin=0.5,vmax=None, pixel_um=13, path_out=path_out)

#%% merge tif and movie
os.makedirs(pjoin(path_out, experiment+'-merge'), exist_ok=True)
for i in range(n_movie):
    movie_name = str(movie_list[i])[2:-6]
    movie_file = pjoin(movie_folder, movie_name + '.mp4')
    tif_file = pjoin(path_out, experiment+'-tif', movie_name + '.tif')
    merge_file = pjoin(path_out, experiment + '-merge', movie_name + '_merged.mp4')
    merge_video(tif_file, movie_file, merge_file, clip=0.08, patches=retino['finalPatchesMarked'], ncol=6,
                text='{}-{} '.format(i + 1, movie_name))
    print('finish merging {}-{}'.format(i + 1, movie_name))

print('finish all merging')


# %%
movie_name_list = []
for i in range(n_movie):
    movie_name_list.append(str(movie_list[i])[2:-6])
#%%
mean_patch_movie = pd.DataFrame(index=retino['finalPatchesMarked'].keys(), columns=movie_name_list)
mean_patch_movie_rep = pd.DataFrame(index=retino['finalPatchesMarked'].keys(), columns=movie_name_list)

for i in range(n_movie):
    print('start processing {}-{}'.format(i+1, str(movie_list[i])[2:-6]))
    tif_imovie = np.tensordot(U, SVTcorr_sort[:, :, i, :], axes=(2, 0)).astype('float32')
    tif_imovie_frame_mean = np.mean(tif_imovie, axis=2)
    tif_imovie_frame_rep_mean = np.mean(tif_imovie_frame_mean, axis=2)
    for j in retino['finalPatchesMarked'].keys():
        patch_npix = np.count_nonzero(retino['finalPatchesMarked'][j].array)
        npix_total = np.sum(retino['finalPatchesMarked'][j].array * tif_imovie_frame_rep_mean)
        mean_patch_movie.loc[j, movie_name_list[i]] = npix_total / patch_npix
        npix_total_rep = np.asmatrix(retino['finalPatchesMarked'][j].array.reshape(-1)) @ tif_imovie_frame_mean.reshape(-1, n_rep)
        mean_patch_movie_rep.loc[j, movie_name_list[i]] = npix_total_rep / patch_npix
print('finish mean_patch_movie')

#%% 画不同patch均值热图
mean_patch_movie.to_csv(pjoin(path_out,'mean_patch_movie.csv'), index=True)
for col in mean_patch_movie.columns:
    mean_patch_movie[col] = pd.to_numeric(mean_patch_movie[col])
means = mean_patch_movie.to_numpy()

mean_list = np.array(mean_patch_movie_rep.to_numpy().tolist())
n_rows, n_cols = means.shape
means_rep = mean_list.reshape(n_rows, n_cols, n_rep)

#%%
plot_heatmap(means, mean_patch_movie.columns, mean_patch_movie.index, vmin=-0.002, vmax=0.007,
             title='mean of repetition and frames', outfile=pjoin(path_out, 'mean_patch_movie.png'))

for i in range(n_rep):
    plot_heatmap(means_rep[:, :, i], mean_patch_movie.columns, mean_patch_movie.index, vmin=-0.002, vmax=0.007,
                 title='rep{} mean of frames'.format(i+1), outfile=pjoin(path_out, 'mean_patch_movie_rep{}.png'.format(i+1)))

# %%
v1copy = np.tile(means[0,:], (7, 1))
means_v1 = means - v1copy

plot_heatmap(means_v1, mean_patch_movie.columns, mean_patch_movie.index, vmin=means_v1.min(), vmax=-means_v1.min(),
             title='mean of repetition and frames - V1', outfile=pjoin(path_out, 'mean_patch_movie-V1.png'))



