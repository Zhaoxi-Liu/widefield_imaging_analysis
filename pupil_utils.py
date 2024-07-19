import numpy as np
import cv2

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


#%%
def merge_pupil_face_video(pupil_video_sort_nmovie, face_video_sort_nmovie, WF_videofile, pupil_mergefile, tif_width, tif_height, text=None):

    WF_video = cv2.VideoCapture(pjoin(WF_videofile))
    fps = WF_video.get(cv2.CAP_PROP_FPS)
    WF_width = int(WF_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    WF_height = int(WF_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    pupil_resize_height = tif_height - face_video_height
    pupil_resize_width = round(pupil_video_sort.shape[4] * pupil_resize_height / pupil_video_sort.shape[3])

    cv2.namedWindow(basename(pupil_mergefile), cv2.WINDOW_NORMAL)
    cv2.resizeWindow(basename(pupil_mergefile), WF_width // 2, WF_height // 2)
    cv2.moveWindow(basename(pupil_mergefile), 1920, 0)

    out = cv2.VideoWriter(pupil_mergefile, cv2.VideoWriter_fourcc(*'mp4v'), fps, (WF_width, WF_height))
    nframe = int(WF_video.get(cv2.CAP_PROP_FRAME_COUNT))
    for iframe in range(nframe):
        ret1, frame1 = WF_video.read()
        if not ret1:
            break

        frame1[:face_video_height, tif_width:tif_width + face_video_width, :] = face_video_sort_nmovie[iframe, imovie,
                                                                                :, :]
        frame_pupil = cv2.resize(pupil_video_sort_nmovie[iframe, imovie, :, :],
                                 (pupil_resize_width, pupil_resize_height))
        frame1[face_video_height:tif_height, tif_width:tif_width + pupil_resize_width, :] = frame_pupil

        cv2.imshow(basename(pupil_mergefile), frame1)
        cv2.waitKey(1)
        # 输出拼接帧
        out.write(frame1)
        print('finish merging {}{}th frame'.format(text, iframe + 1))

    # 释放资源
    WF_video.release()
    out.release()
    cv2.destroyAllWindows()

