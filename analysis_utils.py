from glob import glob
from os.path import join as pjoin
from tifffile import imread
from wf_utils import filename2int, log_progress
from ipywidgets import interact

import matplotlib.pyplot as plt
import numpy as np
import os

def cal_velocity(even_data, measure_interval=1000):
    '''
    Calculate the speed of a motor encoder based on wave_A and wave_B channels
    When the codewheel rotates in the clockwise direction (as viewed from the 
    encoder end of the motor), channel A will lead channel B. If the codewheel 
    rotates in the counterclockwise direction, channel B will lead channel A.
    
    mesure_interval: the interval for measurements the velocity, in ms
    '''

    cpr = 200
    diammeter = 2.54 # cm
    split_index = np.arange(0, even_data.shape[0], measure_interval)
    split_index = np.append(split_index, even_data.shape[0])

    velocity = np.zeros((split_index.size-1))
    for i in range(split_index.size-1):
        wave_A = even_data[split_index[i]:split_index[i+1], 1]
        wave_B = even_data[split_index[i]:split_index[i+1], 2]
        raise_A = np.where(np.diff(wave_A) == 1)[0]
        count = 0
        for j in range(raise_A.size):
            if wave_B[raise_A[j]] == 0:
                count += 1
            else:
                count -= 1
        velocity[i] = (count / cpr * np.pi * diammeter)
        
    return velocity

def correct_lum_outliers(bin_path, outlier_index_470, outlier_index_405, plot=True):
    if (outlier_index_470 is not None) or (outlier_index_405 is not None):
        print('There are luminance outliers frames need to be corrected!')
        bin_name = os.path.splitext(os.path.basename(bin_path))[0]
        dtype = bin_name.split('_')[-1]
        shape = tuple([int(i) for i in bin_name.split('_')[:-1]])
        images = np.memmap(bin_path, dtype=dtype, mode='r+', shape=shape)
        if outlier_index_470 is not None:
            for i in range(outlier_index_470.shape[0]):
                start, end = outlier_index_470
                images[start:end+1, 0, :, :] = (images[start-1, 0, :, :] + images[end+1, 0, :, :]) / 2
        if outlier_index_405 is not None:
            for i in range(outlier_index_405.shape[0]):
                start, end = outlier_index_405
                images[start:end+1, 1, :, :] = (images[start-1, 1, :, :] + images[end+1, 1, :, :]) / 2
        images.flush()
        print('Luminance outliers frames corrected!')
    else:
        print('No luminance outliers frames to correct!')

    if plot:
        print('Checking the corrected images...')
        print('Calulating the mean values of corrected images...')
        mean_values_470 = images[:, 0, :, :].mean(axis=(1,2))
        mean_values_405 = images[:, 1, :, :].mean(axis=(1,2))
        fig, ax = plt.subplots(figsize=(20, 5))
        ax.plot(mean_values_470, label='470', color='r')
        ax.plot(mean_values_405, label='405', color='k')
        ax.legend()
        plt.title('correct_lum_outliers')
        plt.show()

def detect_dim_outliers(mean_values, lum_thr_coef=0.3, plot=True):
    '''
    Detect outliers in the mean_values array.
    There are two types of outliers:
    1. lumiance: the LED not normaly lighted, the mean value is lower than normal.
    2. cross-channel: the 405 or 470 channel frames wrongly recoreded.
    Parameters:
        mean_values: numpy array, the mean values of the image stack. 
        The first column is the mean values of 470 channel, the second column is the mean values of 405 channel.
        lum_thr_coef: luminance threshold coefficient for detecting luminance outliers.
        # diff_thr_coef=0.1, diff_thr_coef: float, the threshold coefficient for detecting cross-channel outliers.
    Returns:
        outlier_index: numpy array, the index of the outliers, 
        each row contains the start and end index of the outliers,
        the value of mean_values[start:end] is considered as outliers.
    '''
    # detecting luminance outliers
    outlier_lum_470 = mean_values[:, 0] < np.mean(mean_values[:, 0])*lum_thr_coef
    outlier_lum_idx_470 = np.where(outlier_lum_470)[0]
    if len(outlier_lum_idx_470) > 0:
        # Frames before and after the detected outliers are also considered as outliers,
        # Output index is the start and end index of the outliers,
        # Note the frame of the end index is normal, so add 2 to the end index.
        outlier_lum_idx_470 = np.array((outlier_lum_idx_470[0]-1, outlier_lum_idx_470[-1]+2))

    outlier_lum_405 = mean_values[:, 1] < np.mean(mean_values[:, 1])*lum_thr_coef
    outlier_lum_idx_405 = np.where(outlier_lum_405)[0]
    if len(outlier_lum_idx_405) > 0:
        outlier_lum_idx_405 = np.array((outlier_lum_idx_405[0]-1, outlier_lum_idx_405[-1]+2))

    outlier_index = [outlier_lum_idx_470, outlier_lum_idx_405]

    if plot:
        fig, ax = plt.subplots(figsize=(20, 5))
        ax.plot(mean_values[:, 0], label='470', color='red')
        ax.plot(mean_values[:, 1], label='405', color='black')
        ax.set_xlim(0, len(mean_values))
        # for idx in outlier_index[0]:
        #     ax.axvline(x=idx, color='r', linestyle='--')
        # for idx in outlier_index[1]:
        #     ax.axvline(x=idx, color='g', linestyle='--')
        # ax.hlines(np.mean(mean_values[:, 0])*lum_thr_coef, 0, len(mean_values), color='r', linestyle='--')
        if len(outlier_lum_idx_470) > 0 or len(outlier_lum_idx_405) > 0:
            ax.hlines(np.mean(mean_values[:, 1])*lum_thr_coef, 0, len(mean_values), color='g', linestyle='--')
        ax.legend()
        plt.title('detect_dim_outliers')

    return outlier_index

def display_wrapper(images, cmap='gray', figsize=(5,5), colorbar=False):
    # max = np.max(images) * 0.85
    # min = np.min(images) + abs(np.min(images) * 0.15)
    def display_images(frame):
    
        plt.figure(figsize=figsize)
        # plt.imshow(images[frame], cmap=cmap, vmax=max, vmin=min)
        plt.imshow(images[frame], cmap=cmap)

        if colorbar:
            plt.colorbar()
        plt.axis('off')
        plt.show()

    frame_slider = interact(display_images, frame=(0, len(images)-1, 1))
    display(frame_slider)

def plot_onset_index(frame_index, title=None):
    frame_interval = np.diff(frame_index)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(frame_interval, marker='o', linestyle='-', color='b')

    # set the y-axis range
    y_range = frame_interval.max() - frame_interval.min()
    y_min = frame_interval.min() - 3*y_range
    y_max = frame_interval.max() + 3*y_range
    ax.set_ylim(bottom=y_min, top=y_max)
    
    ax.set_xlabel('Trial index')
    ax.set_ylabel('Frame interval')
    if title:
        plt.title(title)
    plt.show()

def organize_tif(folder_path, save_path=None):
    folder_name = os.path.basename(folder_path)
    if os.path.exists(folder_path+'.tif'):
        print('importing {}.tif'.format(folder_path))
        image_stack = imread(folder_path+'.tif')
        print('finish importing {}.tif'.format(folder_path))
    else:
        image_path_ls = glob(pjoin(folder_path, '*.tif'))
        image_path_ls = sorted(image_path_ls, key=filename2int) # 确保图像帧按顺序排列
        image_stack = [imread(tiff) for tiff in log_progress(image_path_ls, name=folder_name)]  # 将多帧tif堆叠成数组
        # image_stack = multi_load_images(image_path_ls, n_thread=20)
    # rotated_images = [cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE) for frame in image_stack]   # 顺时针旋转图像90度
    # 计算并保存均值数据
    mean_values = np.array(image_stack).mean(axis=(1,2))
    if save_path is None:
        save_path = folder_path
    output_value = pjoin(save_path, folder_name + "-Values.csv")
    np.savetxt(output_value, mean_values, delimiter=",")
    
    return np.array(image_stack)

def rotate_crop_array(array, angle, left, top, width, height):
    '''
    Rotate the image by the angle and crop the image with the given coordinates
    Args:
        array: numpy array, the image to be rotated and cropped, 2D or 3D. 
        if 3D, the first dimension is the number of images
        angle: float, the rotation angle in degrees. Positive, counterclockwise; Negative, clockwise.
        left: int, the x coordinate of the top-left corner of the cropped image
        top: int, the y coordinate of the top-left corner of the cropped image
        width: int, the width of the cropped image
        height: int, the height of the cropped image
    '''
    if not angle == 0:
        if array.ndim == 2:
            rotated = rotate(array, angle, reshape=False)
        if array.ndim == 3:
            print('Rotating the images...')
            print('This may take a while..., please wait')
            rotated = rotate(array, angle, axes=(2, 1), reshape=False)
        print(rotated.shape)
    else:
        rotated = array

    if left+width > rotated.shape[-2] or top+height > rotated.shape[-1]:
        print(left+width, rotated.shape[-2], top+height, rotated.shape[-1])
        print("The crop area is out of the image")
        return None
    else:
        if rotated.ndim == 3:
            print('Cropping the images...')
            cropped = rotated[:, top:top+height, left:left+width]
        if rotated.ndim == 2:
            cropped = rotated[top:top+height, left:left+width]
        return cropped

def sorting(data, index, seq):
    '''
    Sort the data according to the sequence and event index
    data: 
    index: the index of each onset in the data
    seq: int ndarray, the sequence of stimuli, must be randomized in each repetition,
    and each repetition must contian all the stimuli once.
    '''
    seq = seq.astype(int)
    [n_roi, n_sample] = data.shape
    print('n_roi:', n_roi, 'n_sample:', n_sample)
    n_seq = seq.size
    n_stim = np.unique(seq).size
    n_rep = int(n_seq/n_stim)
    print('n_rep:', n_rep)
    n_index = index.size
    print('n_index:', n_index)
    if n_index != n_stim * n_rep:
        print('n_index != n_stim * n_rep')
    stim_len = np.min(np.diff(index)) # The reason for +1 here is to get enough trace length
    data_sorted = np.zeros((stim_len, n_stim, n_rep+1, n_roi))
    for i_roi in range(n_roi):
        # print('i_roi:', i_roi)
        for i_index in range(n_index):
            # print('i_index:', i_index)
            i_rep = int(i_index/n_stim)
            data_sorted[:, seq[i_index], i_rep, i_roi] = data[i_roi, index[i_index]:index[i_index]+stim_len]
    data_sorted[:,:,n_rep,:] = np.mean(data_sorted[:,:,:n_rep, :],axis=2)
    
    return data_sorted

def sorting_sequence(data, index, seq):
    '''
    Sort the data according to the sequence and event index
    Note: the sequence is ordered, different from the sorting function,
    only use this function to sorting Retinotopy data.
    '''
    seq = seq.astype(int)
    [n_roi, n_sample] = data.shape
    n_seq = seq.size
    n_stim = np.unique(seq).size
    n_rep = int(n_seq/n_stim)
    n_index = index.size
    if n_index != n_stim * n_rep:
        print('n_index != n_stim * n_rep')
    stim_len = np.min(np.diff(index)) # The reason for +1 here is to get enough trace length
    data_sorted = np.zeros((stim_len, n_stim, n_rep+1, n_roi))
    for i_roi in range(n_roi):
        for i_stim in range(n_stim):
            for i_rep in range(n_rep):

                data_sorted[:, i_stim, i_rep, i_roi] = data[i_roi, index[i_stim*n_rep+i_rep]:index[i_stim*n_rep+i_rep]+stim_len]
    data_sorted[:,:,n_rep,:] = np.mean(data_sorted[:,:,:n_rep, :],axis=2)

    return data_sorted
