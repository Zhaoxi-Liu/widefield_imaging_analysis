import NeuroAnalysisTools
import NeuroAnalysisTools.core.FileTools as ft
import NeuroAnalysisTools.RetinotopicMapping as rm
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from os.path import join as pjoin


#%%
def plot_borders(patches, plotAxis=None, title=None, zoom=1,
                 borderWidth=1, isColor=True, plotName=True, fontSize=10):
    import scipy.ndimage as ni
    from NeuroAnalysisTools.core import PlottingTools as pt

    if not plotAxis:
        f = plt.figure(figsize=(10, 10))
        plotAxis = f.add_subplot(111)

    for key, patch in patches.items():
        if isColor:
            if patch.sign == 1:
                plotColor = '#ff0000'
            elif patch.sign == -1:
                plotColor = '#0000ff'
            else:
                plotColor = '#000000'
        else:
            plotColor = '#0000ff'
        currArray = ni.binary_erosion(patch.array, iterations=1)
        im = pt.plot_mask_borders(currArray, plotAxis=plotAxis, color=plotColor, zoom=zoom, borderWidth=borderWidth)
        if plotName:
            center = patch.getCenter()
            plotAxis.text(center[1] * zoom, center[0] * zoom, key, verticalalignment='center',
                          horizontalalignment='center', color=plotColor, fontsize=fontSize)
    plotAxis.set_axis_off()

    if title is not None:
        plotAxis.set_title(title)

    return plotAxis.get_figure()


#%%
def subplot_borders(patches, width=2560, height=512, ncol=5, nrow=1, borderWidth=1, fontSize=10, isColor=False,
                    plotName=True):
    fig, axes = plt.subplots(nrow, ncol, figsize=(width / 100, height / 100), dpi=100)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    for ax in axes:
        plot_borders(patches, plotAxis=ax, zoom=1, borderWidth=borderWidth, fontSize=fontSize,
                     isColor=isColor, plotName=plotName)
    return fig


#%%
def merge_video(tif_file, stim_file, out_file, tif_fps=10, clip=0.05, patches=None):
    import cv2
    import numpy as np
    import tifffile
    import io

    tif_video = tifffile.TiffFile(tif_file)
    mp4_video = cv2.VideoCapture(stim_file)

    width = tif_video.series[0].shape[2]
    height = tif_video.series[0].shape[1] + int(mp4_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    mp4_fps = mp4_video.get(cv2.CAP_PROP_FPS)
    frame_repeat = int(mp4_fps / tif_fps)

    if patches:
        fig = subplot_borders(patches)
        fig.canvas.draw()
        borders_argb = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        borders_rgba = np.copy(borders_argb).view(np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (4,))
        white = np.all(borders_rgba[:, :, :3] == [255, 255, 255], axis=-1)
        borders_rgba[white, 3] = 0

    out = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc(*'mp4v'), mp4_fps, (width, height))
    for i in range(min(len(tif_video.pages) * frame_repeat, int(mp4_video.get(cv2.CAP_PROP_FRAME_COUNT)))):
        j = i // frame_repeat   # 重复tif以适应mp4
        frame1 = tif_video.pages[j].asarray()
        ret2, frame2 = mp4_video.read()
        if not ret2:
            break

        frame1_clipped = np.clip(frame1, -clip, clip)  # 将 frame1 的值限制在 -0.05 到 0.05 的范围内
        frame1_uint8 = ((frame1_clipped + clip) / clip * 127).astype(np.uint8)
        frame1_bgr = cv2.cvtColor(frame1_uint8, cv2.COLOR_GRAY2BGR)

        if patches:
            frame1_rgba = cv2.cvtColor(frame1_uint8, cv2.COLOR_GRAY2RGBA)
            frame1_border = cv2.bitwise_and(frame1_rgba, borders_rgba)
            frame1_bgr = cv2.cvtColor(frame1_border, cv2.COLOR_RGBA2BGR)

        # 给frame2补齐宽度
        pad_width = width - frame2.shape[1]
        frame2_padded = np.pad(frame2, ((0, 0), (0, pad_width), (0, 0)), mode='constant', constant_values=0)

        # 上下拼接帧
        merged_frame = np.concatenate((frame1_bgr, frame2_padded), axis=0)
        # 输出拼接帧
        out.write(merged_frame)

        print('finish merging {}th frame'.format(i))

    # 释放资源
    mp4_video.release()
    out.release()


#%%
tif_file = 'Y:\\WF_VC_liuzhaoxi\\24.3.27_C92\\natural-movie\\process\\20240327-183009-wfield\\..\\20240327-183009-natural-movie\\tif\\forest.tif'
movie_file = 'Y:\\WF_VC_liuzhaoxi\\24.3.27_C92\\natural-movie\\process\\20240327-183009-wfield\\..\\..\\raw\\natural_movies\\forest.mp4'
out_file = 'Y:\\WF_VC_liuzhaoxi\\24.3.27_C92\\natural-movie\\process\\20240327-183009-wfield\\..\\20240327-183009-natural-movie\\merge\\forest_merged.mp4'
with open(r'Y:\WF_VC_liuzhaoxi\24.3.27_C92\retinotopy\process\20240327-200819-retinotopy\retinotopy_out.pkl', 'rb') as f:
    retino = pickle.load(f)
merge_video(tif_file, movie_file, out_file, clip=0.08, patches=retino['finalPatchesMarked'])

#%%
# movie_folder = r'Y:\WF_VC_liuzhaoxi\24.01.03_C83\natural-movie\raw\natural_movies'
# movie_list = pd.read_csv(pjoin(movie_folder, 'movie_list.txt'), header=None).values
# path_out = r'Y:\WF_VC_liuzhaoxi\24.01.03_C83\natural-movie\process\20240103-131221-natural-movie'
# n_movie = movie_list.size
#
# for i in range(n_movie):
#     movie_name = str(movie_list[i])[2:-6]
#     movie_file = pjoin(movie_folder, movie_name + '.mp4')
#     tif_file = pjoin(path_out, 'tif', movie_name + '.tif')
#     merge_file = pjoin(path_out, 'merge', movie_name + '_merged.mp4')
#     merge_video(tif_file, movie_file, merge_file, clip=0.08)
#     print('finish merging {}-{}'.format(i + 1, movie_name))
#
# print('finish all merging')


#%%
'''def merge_video(tif_file, stim_file, merge_file, tif_fps=10, clip=0.05, patches=None):
    import cv2
    import numpy as np
    import tifffile
    import matplotlib.pyplot as plt
    import io

    tif_video = tifffile.TiffFile(tif_file)
    mp4_video = cv2.VideoCapture(stim_file)

    # 获取视频的帧率和尺寸
    mp4_fps = mp4_video.get(cv2.CAP_PROP_FPS)
    frame_repeat = int(mp4_fps / tif_fps)
    width = tif_video.series[0].shape[2]
    height = tif_video.series[0].shape[1] + int(mp4_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 创建一个新的视频写入对象
    out = cv2.VideoWriter(merge_file, cv2.VideoWriter_fourcc(*'mp4v'), mp4_fps, (width, height))

    # 重复帧以达到 60fps
    for i in range(min(len(tif_video.pages) * frame_repeat, int(mp4_video.get(cv2.CAP_PROP_FRAME_COUNT)))):
        j = i // frame_repeat
        frame1 = tif_video.pages[j].asarray()
        ret2, frame2 = mp4_video.read()
        if not ret2:
            break

        # 将 frame1 的值限制在 -0.05 到 0.05 的范围内
        frame1_clipped = np.clip(frame1, -clip, clip)
        # 将 frame1_clipped 转换到 0 到 255 的范围
        frame1_uint8 = ((frame1_clipped + clip) / clip * 127).astype(np.uint8)
        # 叠加轮廓使用 matplotlib 来画图，所以需要将 frame 转换成 RGB 格式
        if patches:
            frame1_rgb = cv2.cvtColor(frame1_uint8, cv2.COLOR_GRAY2RGB)
            plt.figure(figsize=(frame1_rgb.shape[1] / 100, frame1_rgb.shape[0] / 100), dpi=100)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.imshow(frame1_rgb)
            plt.axis('off')
            # plt.savefig('temp1.png')
            ax = plt.gca()
            # 在 frame1_mapped_color 上叠加轮廓
            plot_borders(patches, plotAxis=ax, zoom=1, borderWidth=1, fontSize=8,
                         isColor=True, plotName=True)
            # 获取当前轮廓叠加后的图像
            # buf = io.BytesIO()
            # plt.savefig(buf, format='png', pad_inches=0, dpi=100)
            # buf.seek(0)
            # frame1_bgr = cv2.imdecode(np.frombuffer(buf.read(), np.uint8), 1)
            plt.savefig('temp.png', bbox_inches='tight', pad_inches=0, dpi=100)  # 保存临时图像
            temp_img = cv2.imread('temp.png')
            frame1_bgr = cv2.cvtColor(temp_img, cv2.COLOR_RGB2BGR)
            plt.close()  # 关闭图像，以便下一帧重新绘制
        else:
            frame1_bgr = cv2.cvtColor(frame1_uint8, cv2.COLOR_GRAY2BGR)

        # 计算frame2需要补零的宽度
        pad_width = width - frame2.shape[1]
        # 在frame2右侧补零
        frame2_padded = np.pad(frame2, ((0, 0), (0, pad_width), (0, 0)), mode='constant', constant_values=0)

        # 将两个帧上下拼接在一起
        merged_frame = np.concatenate((frame1_bgr, frame2_padded), axis=0)
        # 输出合并视频
        out.write(merged_frame)
        print('finish merging {}th frame'.format(i))

    # 释放资源
    mp4_video.release()
    out.release()
'''
