import pickle
import numpy as np
import cv2
import os
from os.path import join as pjoin
import pandas as pd

#%%
def read_video(video_path):
    # 读取MP4文件
    cap = cv2.VideoCapture(video_path)
    # 检查视频是否成功打开
    if not cap.isOpened():
        print("Error opening video {}".format(video_path))
    # 读取视频帧直到结束
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    # 释放视频流并关闭窗口
    cap.release()

    return frames


#%%
def sorting_NatMov_pupil(pupil, trials_onset, n_movie, n_frame):
    shape = pupil.shape
    pupil_reshape = pupil.reshape(shape[0], -1)
    dtype = pupil.dtype

    n_trials = trials_onset.size
    n_rep = n_trials // n_movie
    pupil_reshape_sorted = np.zeros(shape=(n_frame, n_movie, n_rep, pupil_reshape.shape[-1]), dtype=dtype)
    for i_trial in range(n_trials):
        i_movie = i_trial // n_rep
        i_rep = i_trial % n_rep
        pupil_reshape_sorted[:n_frame, i_movie, i_rep, :] = pupil_reshape[trials_onset[i_trial]:trials_onset[i_trial] + n_frame, :]
    pupil_sorted = pupil_reshape_sorted.reshape(n_frame, n_movie, n_rep, *shape[1:]).squeeze()

    return pupil_sorted


# %% set path and parameter
path_wfield = r'Y:\WF_VC_liuzhaoxi\24.3.27_C92\natural-movie\process\20240327-183009-wfield'
path_retinotopy = r'Y:\WF_VC_liuzhaoxi\24.3.27_C92\retinotopy\process\20240327-200819-retinotopy'# path_retinotopy = r'Y:\WF_VC_liuzhaoxi\24.01.03_C83\retinotopy\process\20240103-123936-retinotopy'

experiment = os.path.basename(path_wfield)[:15]
rawPath = pjoin(path_wfield, '..\\..\\raw')
path_out = pjoin(path_wfield, '..', experiment + '-natural-movie')
os.makedirs(path_out, exist_ok=True)
movie_folder = pjoin(rawPath,'natural_movies')
movie_list = pd.read_csv(pjoin(movie_folder, 'movie_list.txt'), header=None).values
n_movie = movie_list.size
trialfile = pd.read_csv(pjoin(path_wfield, 'trials.csv'), header=None).values.astype(int)

n_frame = 150  # 帧


#%%
pupil_video_path = r'Y:\WF_VC_liuzhaoxi\24.3.27_C92\natural-movie\process\20240327-183009-pupil\20240327-183009-event-sum_pupil_datection_l130_r170_t30_b65.mp4'
pupil_data_path = r'Y:\WF_VC_liuzhaoxi\24.3.27_C92\natural-movie\process\20240327-183009-pupil\20240327-183009-event-sum_pupil_detection.pkl'

with open(pupil_data_path, 'rb') as f:
    pupil_data = pickle.load(f)
f.close()
pupil_area_sort=sorting_NatMov_pupil(pupil_data['Pupil_area'], trialfile[:, 1], n_movie, n_frame)

pupil_video = read_video(pupil_video_path)
pupil_video_sort = sorting_NatMov_pupil(np.asarray(pupil_video), trialfile[:, 1], n_movie, n_frame)