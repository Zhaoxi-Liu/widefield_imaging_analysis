from os.path import join as pjoin
import numpy as np
import matplotlib.pyplot as plt


# %%
def sorting_NatMov(SVT, trials_onset, n_movie, movie_len):
    nSVD = SVT.shape[0]
    n_trials = trials_onset.size
    n_rep = n_trials // n_movie

    SVT_sorted = np.zeros((nSVD, movie_len, n_movie, n_rep))
    for i_trial in range(n_trials):
        i_movie = i_trial // n_rep
        i_rep = i_trial % n_rep
        SVT_sorted[:nSVD, :movie_len, i_movie, i_rep] = SVT[:, trials_onset[i_trial]:trials_onset[i_trial] + movie_len]

    return SVT_sorted


#%%
def cal_snr(x, axis1, axis2):
    """
    axis1:重复，axis2:时间。重复axis要在时间axis之后！
    """
    snr = np.nanvar(np.nanmean(x, axis=axis1), axis=axis2) / np.nanmean(np.nanvar(x, axis=axis1), axis=axis2)
    return snr


# %%
'''
def plot_movie_snr_(snr, n_rows, n_cols, movie_list, path_out=None):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 12))
    vmin = np.min(snr)
    vmax = np.max(snr)

    for i in range(n_rows):
        for j in range(n_cols):
            index = i * n_cols + j
            if index < snr.shape[2]:
                img = axes[i, j].imshow(snr[:, :, index], cmap='hot', interpolation='nearest', vmin=vmin,
                                        vmax=vmax)
                axes[i, j].set_title(str(movie_list[index])[2:-6])
                cbar = fig.colorbar(img, ax=axes[i, j], shrink=0.8)
            axes[i, j].axis('off')

    fig.set_facecolor('white')
    plt.tight_layout()
    if path_out is not None:
        plt.savefig(pjoin(path_out, 'snr_per_movie.png'), bbox_inches='tight')
    plt.show()
'''


# %%
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
                plotColor = '#0000ff'
            elif patch.sign == -1:
                plotColor = '#0000ff'
            else:
                plotColor = '#000000'
        else:
            plotColor = 'white'
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


# %%
def plot_movie_snr(snr, n_rows, n_cols, movie_list, path_out=None, vmin=None, vmax=None, pixel_um=None, patches=None):
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(n_rows, n_cols + 1, width_ratios=[1] * n_cols + [0.05])  # 添加一列给colorbar
    if vmin is None:
        vmin = np.min(snr)
    if vmax is None:
        vmax = np.max(snr)

    for i in range(n_rows):
        for j in range(n_cols):
            index = i * n_cols + j
            if index < snr.shape[2]:
                ax = fig.add_subplot(gs[i, j])
                img = ax.imshow(snr[:, :, index], cmap='hot', interpolation='nearest', vmin=vmin, vmax=vmax)
                if pixel_um is not None:
                    # Calculate the length of the scale line in pixels
                    scale_line_length = 1e3 / pixel_um
                    line = plt.Line2D([0, 0 + scale_line_length], [snr.shape[1] - 4, snr.shape[1] - 4], color='white',
                                      linewidth=2)
                    plt.gca().add_line(line)
                    plt.text(scale_line_length / 2, snr.shape[1] + 15, '1mm', color='black', fontsize=12, ha='center')
                if patches:
                    # 调用plot_borders函数在当前轴上画边界
                    plot_borders(patches, plotAxis=ax, title=None, zoom=1,
                                 borderWidth=1, isColor=False, plotName=True, fontSize=8)

                ax.set_title(str(movie_list[index])[2:-6])
                ax.axis('off')

    cbar_ax = fig.add_subplot(gs[:, -1])  # 在最后一列添加colorbar位置
    cbar = fig.colorbar(img, cax=cbar_ax)
    fig.set_facecolor('white')
    plt.tight_layout()
    if path_out is not None:
        plt.savefig(pjoin(path_out, 'snr_per_movie.png'), bbox_inches='tight')
    plt.show()


# %%
def subplot_borders(patches, width=2560, height=512, ncol=5, nrow=1, borderWidth=1, fontSize=10, isColor=True,
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


# %%
def merge_video(tif_file, stim_file, out_file, tif_fps=10, clip=0.05, patches=None, ncol=5, nrow=1, text=''):
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
        fig = subplot_borders(patches, ncol=ncol, nrow=nrow, width=width, height=tif_video.series[0].shape[1])
        fig.canvas.draw()
        borders_argb = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        borders_rgba = np.copy(borders_argb).view(np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (4,))
        white = np.all(borders_rgba[:, :, :3] == [255, 255, 255], axis=-1)
        borders_rgba[white, 3] = 0

    out = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc(*'mp4v'), mp4_fps, (width, height))
    for i in range(min(len(tif_video.pages) * frame_repeat, int(mp4_video.get(cv2.CAP_PROP_FRAME_COUNT)))):
        j = i // frame_repeat  # 重复tif以适应mp4
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

        print('finish merging {}{}th frame'.format(text, i))

    # 释放资源
    mp4_video.release()
    out.release()


#%%
def plot_heatmap(data, xlable, ylable, cmap='coolwarm', vmin=None, vmax=None, title=None, outfile=None, dpi=300):
    n_rows, n_cols = data.shape
    if vmin == None:
        vmin = data.min()
    if vmax == None:
        vmax = data.max()
    plt.figure(figsize=(16, 8))
    plt.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, extent=[0, n_cols, 0, n_rows])
    plt.colorbar()
    plt.xticks(np.arange(n_cols) + 0.5, xlable, rotation=45, ha='right')
    plt.yticks(np.arange(n_rows) + 0.5, ylable[::-1])

    # 在每个格子中添加数值
    for i in range(n_rows):
        for j in range(n_cols):
            plt.text(j + 0.5, n_rows - i - 0.5, '{:.6f}'.format(data[i, j]), color='black',
                     ha='center', va='center', fontsize=8)
    plt.gca().invert_yaxis()  # 倒置y轴
    if title:
        plt.title(title)
    if outfile:
        plt.savefig(outfile, bbox_inches='tight', dpi=dpi)
    plt.show()
