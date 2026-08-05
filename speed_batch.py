#%%
import os
from os.path import join as pjoin
from tifffile import imread, imwrite, TiffFile
import numpy as np
import shutil
import matplotlib.pyplot as plt
from glob import glob
import pandas as pd
import cv2
from tqdm import tqdm
import subprocess
from wf_utils import *
from wfield_utils import *
# from multi_load_images import *

#%%
path_ls=glob("/home/lzx/data/WF_VC_liuzhaoxi/25.*_*")


#%%
for path_0 in path_ls:

    # path_0: including one-day recording files
    # path_1: including every stimuli
    path_0 = r"/home/lzx/data/WF_VC_liuzhaoxi/25.08.07_A11"
    path_1_ = os.listdir(path_0)
    # 筛选出文件夹
    path_1 = [folder for folder in path_1_ if os.path.isdir(pjoin(path_0, folder))]
    path_all = []
    for path in path_1:
        path_all.append(pjoin(path_0, path))

    print('All the file path:\n'+'\n'.join(path_all))


    # 列出所有已经合并通道的tiff的路径

    path_wfield_all = glob(pjoin(path_0, '*/process/*-wfield'))
    print('All the wfield-path:\n'+"\n".join(path_wfield_all))


    ### 生成speed.csv文件

    for path_wfield in path_wfield_all:
        
        experiment = os.path.basename(path_wfield)[:15]
        rawPath = pjoin(path_wfield,'../../raw')
        merge_file = pjoin(path_wfield, experiment+"-merged.tif")
        n_frames = len(TiffFile(merge_file).pages)//2

        if os.path.exists(pjoin(rawPath, experiment + ".csv")):
            stimfile = pd.read_csv(pjoin(rawPath, experiment + ".csv"), header=None).values
            speed_raw = stimfile[:, 1:3].astype(int)  # columns: A, B
            stim_delay = pd.read_csv(pjoin(rawPath, experiment + "-470Timestamp.csv"), header=None).values
            stim_delay = int(stim_delay[0, 0])

            # === params ===
            fs = 1000          # Hz (sampling rate)
            fps = 10           # Hz (frame rate)
            samples_per_frame = fs // fps  # 100
            numGratings = 200.0
            wheel_diameter = 2.54
            PI = np.pi
        
            # number of frames that can be computed from available samples
            n_stimframe = min((speed_raw.shape[0] - stim_delay) // samples_per_frame, n_frames)
            speed = np.zeros(n_stimframe, dtype=float)

            def count_quadrature_events(A: np.ndarray, B: np.ndarray) -> float:
                """
                Returns signed 'grating count' for the window (clock - counterClock),
                where each direction uses avg(start, end) as in the original code.
                """
                # previous and current
                Al, Ac = A[:-1], A[1:]
                Bl, Bc = B[:-1], B[1:]

                # A rising: Al=0, Ac=1
                Arise = (Al == 0) & (Ac == 1)
                # A falling: Al=1, Ac=0
                Afall = (Al == 1) & (Ac == 0)

                # Arduino definitions:
                # clockStart:  A rise AND B stays 0 (Bl=0,Bc=0)
                clockStart = Arise & (Bl == 0) & (Bc == 0)
                # clockEnd:    A fall AND B stays 1 (Bl=1,Bc=1)
                clockEnd = Afall & (Bl == 1) & (Bc == 1)

                # counterClockStart: A rise AND B stays 1
                ccStart = Arise & (Bl == 1) & (Bc == 1)
                # counterClockEnd:   A fall AND B stays 0
                ccEnd = Afall & (Bl == 0) & (Bc == 0)

                clock_cnt = 0.5 * (clockStart.sum() + clockEnd.sum())
                cc_cnt = 0.5 * (ccStart.sum() + ccEnd.sum())
                return float(clock_cnt - cc_cnt)

            def grating_to_speed(cnt_grating: float, window_sec: float) -> float:
                distance = wheel_diameter * PI * (cnt_grating / numGratings)
                return distance / window_sec

            window_sec = 1.0 / fps  # 0.1s

            for i in range(n_stimframe):
                s0 = stim_delay + i * samples_per_frame
                s1 = s0 + samples_per_frame

                A = speed_raw[s0:s1, 0]
                B = speed_raw[s0:s1, 1]

                # If the window is too short (shouldn't happen), keep 0
                if A.size < 2:
                    speed[i] = 0.0
                    continue

                cnt = count_quadrature_events(A, B)
                speed[i] = grating_to_speed(cnt, window_sec)

            # === save ===
            out_df = pd.DataFrame({"frame": np.arange(n_stimframe), "speed": speed})
            out_path = pjoin(path_wfield, experiment+"-speed.csv")
            out_df.to_csv(out_path, index=False)

            print("generate speed.csv of "+experiment)
            

        else: print(experiment, 'without stim-file, skip')
                
    print("\nAll generation of speed.csv finished!")



# %%
