from os.path import join as pjoin
import numpy as np
import matplotlib.pyplot as plt
from os.path import basename as basename

# %%
def sorting_NatMov(SVT, trials_onset, n_movie, movie_len, pre_length=0, after_length=0):
    # SVT: [ncomponent, n_frame]
    nSVD = SVT.shape[0]
    n_trials = trials_onset.size
    n_rep = n_trials // n_movie

    SVT_sorted = np.empty((nSVD, movie_len + pre_length + after_length, n_movie, n_rep))
    for i_trial in range(n_trials):
        i_movie = i_trial // n_rep
        i_rep = i_trial % n_rep
        SVT_sorted[:nSVD, :movie_len + pre_length + after_length, i_movie, i_rep] = SVT[:,
                                                                                  trials_onset[i_trial] - pre_length:
                                                                                  trials_onset[
                                                                                      i_trial] + movie_len + after_length]

    return SVT_sorted


# %%
def cal_snr(x, axis1, axis2):
    """
    axis1:重复，axis2:时间。重复axis要在时间axis之后！
    """
    snr = np.nanvar(np.nanmean(x, axis=axis1), axis=axis2) / np.nanmean(np.nanvar(x, axis=axis1), axis=axis2)
    return snr


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
def subplot_movie_heatmap_(movie_data, n_rows, n_cols, movie_name_list, path_outfile=None, title=None, vmin=None, vmax=None, cmap='hot',
                    pixel_um=None, patches=None, ccf_regions=None, ccf_color='w', dpi=200):
    if vmin is None:
        vmin = np.min(movie_data)
    if vmax is None:
        vmax = np.max(movie_data)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 12))
    for i in range(n_rows):
        for j in range(n_cols):
            index = i * n_cols + j
            if index < movie_data.shape[2]:
                img = axes[i, j].imshow(movie_data[:, :, index], cmap=cmap, interpolation='nearest', vmin=vmin,
                                        vmax=vmax)
                axes[i, j].set_title(movie_name_list[index], pad=0)
                axes[i, j].title.set_position([0.5, 1.00])
                cbar = fig.colorbar(img, ax=axes[i, j])
                if pixel_um is not None:
                    # Calculate the length of the scale line in pixels
                    scale_line_length = 1e3 / pixel_um
                    line = plt.Line2D([0, 0 + scale_line_length], [movie_data.shape[1] - 4, movie_data.shape[1] - 4],
                                      color='white',
                                      linewidth=2)
                    axes[i, j].add_line(line)
                    axes[i, j].text(scale_line_length / 2, movie_data.shape[1] + 25, '1mm', color='black', fontsize=10,
                             ha='center')
                if patches:
                    # 调用plot_borders函数在当前轴上画边界
                    plot_borders(patches, plotAxis=axes[i, j], title=None, zoom=1,
                                 borderWidth=1, isColor=False, plotName=True, fontSize=8)
                if ccf_regions is not None:
                    for idx, r in ccf_regions.iterrows():
                        axes[i, j].plot(r['left_x'], r['left_y'], ccf_color, lw=0.2)
                        axes[i, j].plot(r['right_x'], r['right_y'], ccf_color, lw=0.2)
                        axes[i, j].text(r.left_center[0], r.left_center[1], r.acronym, color=ccf_color, va='center', fontsize=4, alpha=1, ha='center')
            axes[i, j].axis('off')

    fig.set_facecolor('white')
    if title:
        fig.suptitle(title, fontsize=24)
    plt.tight_layout(h_pad=0.5, w_pad=0, rect=[0, 0, 1, 0.97])
    if path_outfile is not None:
        plt.savefig(path_outfile, bbox_inches='tight', dpi=dpi)
    plt.show()


# %%
def subplot_movie_heatmap(movie_data, n_rows, n_cols, movie_name_list, path_outfile=None, title=None, vmin=None, vmax=None,
                          cmap='hot', pixel_um=None, patches=None, ccf_regions=None, ccf_color='w'):
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(n_rows, n_cols + 1, width_ratios=[1] * n_cols + [0.05])  # 添加一列给colorbar
    if vmin is None:
        vmin = np.min(movie_data)
    if vmax is None:
        vmax = np.max(movie_data)

    for i in range(n_rows):
        for j in range(n_cols):
            index = i * n_cols + j
            if index < movie_data.shape[2]:
                ax = fig.add_subplot(gs[i, j])
                img = ax.imshow(movie_data[:, :, index], cmap=cmap, interpolation='nearest', vmin=vmin, vmax=vmax)
                if pixel_um is not None:
                    # Calculate the length of the scale line in pixels
                    scale_line_length = 1e3 / pixel_um
                    line = plt.Line2D([0, 0 + scale_line_length], [movie_data.shape[1] - 4, movie_data.shape[1] - 4], color='white',
                                      linewidth=2)
                    plt.gca().add_line(line)
                    plt.text(scale_line_length / 2, movie_data.shape[1] + 15, '1mm', color='black', fontsize=12, ha='center')
                if patches:
                    # 调用plot_borders函数在当前轴上画边界
                    plot_borders(patches, plotAxis=ax, title=None, zoom=1,
                                 borderWidth=1, isColor=False, plotName=True, fontSize=8)
                if ccf_regions is not None:
                    for idx, r in ccf_regions.iterrows():
                        ax.plot(r['left_x'], r['left_y'], ccf_color, lw=0.2)
                        ax.plot(r['right_x'], r['right_y'], ccf_color, lw=0.2)
                        ax.text(r.left_center[0], r.left_center[1], r.acronym, color=ccf_color, va='center', fontsize=4, alpha=1, ha='center')

                ax.set_title(movie_name_list[index], pad=0)
                ax.title.set_position([0.5, 1.00])
                ax.axis('off')

    cbar_ax = fig.add_subplot(gs[:, -1])  # 在最后一列添加colorbar位置
    cbar = fig.colorbar(img, cax=cbar_ax)
    if title:
        fig.suptitle(title, fontsize=24)
    fig.set_facecolor('white')
    plt.tight_layout(h_pad=0.5, w_pad=0, rect=[0, 0, 1, 0.97])
    if path_outfile is not None:
        plt.savefig(path_outfile, bbox_inches='tight')
    plt.show()


# %%
def subplot_borders(patches, width=2560, height=512, ncol=1, nrow=1, borderWidth=0.5, fontSize=5, isColor=True,
                    plotName=True):
    fig, axes = plt.subplots(nrow, ncol, figsize=(width / 200, height / 200), dpi=200)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)

    if ncol == 1 and nrow == 1:
        plot_borders(patches, plotAxis=axes, zoom=1, borderWidth=borderWidth, fontSize=fontSize,
                     isColor=isColor, plotName=plotName)
    elif ncol > 1 and nrow > 1:
        for irow in range(nrow):
            for icol in range(ncol):
                ax = axes[irow, icol]
                plot_borders(patches, plotAxis=ax, zoom=1, borderWidth=borderWidth, fontSize=fontSize,
                             isColor=isColor, plotName=plotName)
    else:
        for ax in axes:
            plot_borders(patches, plotAxis=ax, zoom=1, borderWidth=borderWidth, fontSize=fontSize,
                         isColor=isColor, plotName=plotName)

    return fig


# %%
def merge_patch_stim(out_file, tif_file, stim_file=None, tif_fps=10, clip=0.05, patches=None, ncol=1, nrow=1, trial_rep=1, text='', reverse=False):
    import cv2
    import numpy as np
    import tifffile

    tif_video = tifffile.TiffFile(tif_file)
    tif_width = tif_video.series[0].shape[2]
    tif_height = tif_video.series[0].shape[1]

    mp4_video = cv2.VideoCapture(stim_file)
    mp4_width = int(mp4_video.get(cv2.CAP_PROP_FRAME_WIDTH) / 2)
    mp4_height = int(mp4_video.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2)

    width = max(tif_width, mp4_width)
    height = tif_height + mp4_height
    mp4_fps = mp4_video.get(cv2.CAP_PROP_FPS)
    frame_repeat = round(mp4_fps / tif_fps)

    if patches:
        fig = subplot_borders(patches, width=tif_width, height=tif_height, ncol=ncol, nrow=nrow)
        fig.canvas.draw()
        borders_argb = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        borders_rgba = np.copy(borders_argb).view(np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (4,))
        white = np.all(borders_rgba[:, :, :3] == [255, 255, 255], axis=-1)
        borders_rgba[white, 3] = 0

    out = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc(*'mp4v'), tif_fps, (width, height))
    if not out.isOpened():
        print("can't open output video file")
        return

    cv2.namedWindow(basename(out_file), cv2.WINDOW_NORMAL)
    cv2.resizeWindow(basename(out_file), width // 2, height // 2)
    cv2.moveWindow(basename(out_file), 1920, 0)

    for iframe in range(len(tif_video.pages)):
        frame1 = tif_video.pages[iframe].asarray()
        frame1_clipped = np.clip(frame1, -clip, clip)
        frame1_uint8 = ((frame1_clipped + clip) / clip * 127).astype(np.uint8)
        frame1_bgr = cv2.cvtColor(frame1_uint8, cv2.COLOR_GRAY2BGR)

        if patches:
            frame1_rgba = cv2.cvtColor(frame1_uint8, cv2.COLOR_GRAY2RGBA)
            frame1_border = cv2.bitwise_and(frame1_rgba, borders_rgba)
            frame1_bgr = cv2.cvtColor(frame1_border, cv2.COLOR_RGBA2BGR)

        if stim_file:
            nframe_trial = round(len(tif_video.pages) / trial_rep)
            mp4_video.set(cv2.CAP_PROP_POS_FRAMES, (iframe % nframe_trial) * frame_repeat)
            ret2, frame2 = mp4_video.read()
            if not ret2:
                continue
            if reverse:
                # 上下颠倒视频帧
                frame2 = cv2.flip(frame2, 0)
            # 缩小 natural movie 视频帧的大小
            frame2_resized = cv2.resize(frame2, (mp4_width, mp4_height))
            # 补齐宽度
            if tif_width > mp4_width:
                pad_width = width - mp4_width
                frame2_resized = np.pad(frame2_resized, ((0, 0), (0, pad_width), (0, 0)), mode='constant',
                                        constant_values=0)
            else:
                pad_width = width - tif_width
                frame1_bgr = np.pad(frame1_bgr, ((0, 0), (0, pad_width), (0, 0)), mode='constant', constant_values=0)
            # 上下拼接帧
            merged_frame = np.concatenate((frame1_bgr, frame2_resized), axis=0)
        else:
            merged_frame = frame1_bgr
        cv2.imshow(basename(out_file), merged_frame)
        cv2.waitKey(1)

        # 输出拼接帧
        out.write(merged_frame)
        print('finish merging {}{}th frame'.format(text, iframe + 1))

    # 释放资源
    mp4_video.release()
    out.release()
    cv2.destroyAllWindows()
    print('finish merging all frames of {}'.format(tif_file))


# %%
def merge_ccf_stim(out_file, tif_file, ccf_regions, stim_file=None, ncol=1, nrow=1,
                   tif_fps=10, vmin=None, vmax=None, trial_rep=1, text='', reverse=False):
    import numpy as np
    import cv2
    from tifffile import TiffFile


    tif_video = TiffFile(tif_file)
    tif_width = tif_video.pages[0].shape[1]
    tif_height = tif_video.pages[0].shape[0]

    mp4_video = cv2.VideoCapture(stim_file)
    mp4_width = int(mp4_video.get(cv2.CAP_PROP_FRAME_WIDTH) / 2)
    mp4_height = int(mp4_video.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2)

    width = max(tif_width, mp4_width)
    height = tif_height + mp4_height
    mp4_fps = mp4_video.get(cv2.CAP_PROP_FPS)
    frame_repeat = round(mp4_fps / tif_fps)

    out = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc(*'mp4v'), tif_fps, (width, height))
    if not out.isOpened():
        print("can't open output video file")
        return

    if vmin is None:
        vmin = tif_video.pages[0].asarray().min()
    if vmax is None:
        vmax = tif_video.pages[0].asarray().max()

    cv2.namedWindow(basename(out_file), cv2.WINDOW_NORMAL)
    cv2.resizeWindow(basename(out_file), width // 2, height // 2)
    cv2.moveWindow(basename(out_file), 1920, 0)

    for iframe in range(len(tif_video.pages)):
        # plot iframe with allen map
        fig = plt.figure(figsize=(tif_width/128, tif_height/128), dpi=128)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.imshow(tif_video.pages[iframe].asarray(), clim=[vmin, vmax], cmap='gray')
        for irow in range(nrow):
            for icol in range(ncol):
                    for idx, r in ccf_regions.iterrows():
                        plt.plot(np.array(r['left_x'])+icol*(tif_width/ncol), np.array(r['left_y'])+irow*(tif_height/nrow), 'r', lw=0.3)
                        plt.plot(np.array(r['right_x'])+icol*(tif_width/ncol), np.array(r['right_y'])+irow*(tif_height/nrow), 'r', lw=0.3)
                        plt.text(r.left_center[0], r.left_center[1], r.acronym, color='w', va='center', fontsize=6, alpha=0.5, ha='center')
        plt.axis('off')
        fig.set_facecolor('white')

        # export to mp4 video
        fig.canvas.draw()
        frame1_argb = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        frame1_rgba = np.roll(frame1_argb.reshape(fig.canvas.get_width_height()[::-1] + (4,)), 3, axis=2)
        frame1_bgr = cv2.cvtColor(frame1_rgba, cv2.COLOR_RGBA2BGR)
        plt.close(fig)

        if frame1_bgr is None:
            print('帧数据为空：第 {} 帧'.format(iframe + 1))
            print('error when merging {}.mp4'.format(tif_file))
            break

        if frame1_bgr.shape[1] != tif_width or frame1_bgr.shape[0] != tif_height:
            print('帧大小不匹配：第 {} 帧，大小 {}'.format(iframe + 1, frame1_bgr.shape))
            # 调整帧大小
            frame1_bgr = cv2.resize(frame1_bgr, (tif_width, tif_height), interpolation=cv2.INTER_LINEAR)

        if stim_file:
            nframe_trial = round(len(tif_video.pages) / trial_rep)
            mp4_video.set(cv2.CAP_PROP_POS_FRAMES, (iframe % nframe_trial) * frame_repeat)
            ret2, frame2 = mp4_video.read()
            if not ret2:
                continue
            if reverse:
                # 上下颠倒视频帧
                frame2 = cv2.flip(frame2, 0)
            # 缩小 natural movie 视频帧的大小
            frame2_resized = cv2.resize(frame2, (mp4_width, mp4_height))
            # 补齐宽度
            if tif_width > mp4_width:
                pad_width = width - mp4_width
                frame2_resized = np.pad(frame2_resized, ((0, 0), (0, pad_width), (0, 0)), mode='constant',
                                        constant_values=0)
            else:
                pad_width = width - tif_width
                frame1_bgr = np.pad(frame1_bgr, ((0, 0), (0, pad_width), (0, 0)), mode='constant', constant_values=0)
            # 上下拼接帧
            merged_frame = np.concatenate((frame1_bgr, frame2_resized), axis=0)
        else:
            merged_frame = frame1_bgr
        cv2.imshow(basename(out_file), merged_frame)
        cv2.waitKey(1)

        # 写入帧到视频
        out.write(merged_frame)
        print('finish merging {}{}th frame'.format(text, iframe + 1))

    mp4_video.release()
    out.release()
    cv2.destroyAllWindows()
    print('finish merging all frames of {}'.format(tif_file))


# %%
def merge2video(out_file, tif_videofile, stim_file, text=''):
    import cv2
    import numpy as np

    tif_video = cv2.VideoCapture(tif_videofile)
    tif_width = int(tif_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    tif_height = int(tif_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tif_fps = tif_video.get(cv2.CAP_PROP_FPS)

    stim_video = cv2.VideoCapture(stim_file)
    stim_width = int(stim_video.get(cv2.CAP_PROP_FRAME_WIDTH) / 2)
    stim_height = int(stim_video.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2)
    stim_fps = stim_video.get(cv2.CAP_PROP_FPS)

    width = max(tif_width, stim_width)
    height = tif_height + stim_height
    frame_repeat = round(stim_fps / tif_fps)

    out = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc(*'mp4v'), tif_fps, (width, height))

    n_frame = min(int(tif_video.get(cv2.CAP_PROP_FRAME_COUNT)),
                  int(stim_video.get(cv2.CAP_PROP_FRAME_COUNT)) // frame_repeat)
    for iframe in range(n_frame):
        ret1, frame1 = tif_video.read()
        if not ret1:
            break

        stim_video.set(cv2.CAP_PROP_POS_FRAMES, iframe * frame_repeat)
        ret2, frame2 = stim_video.read()
        if not ret2:
            break

        # 缩小 natural movie 视频帧的大小
        frame2_resized = cv2.resize(frame2, (stim_width, stim_height))
        # 补齐宽度
        if tif_width > stim_width:
            pad_width = width - stim_width
            frame2_resized = np.pad(frame2_resized, ((0, 0), (0, pad_width), (0, 0)), mode='constant',
                                    constant_values=0)
        else:
            pad_width = width - tif_width
            frame1 = np.pad(frame1, ((0, 0), (0, pad_width), (0, 0)), mode='constant', constant_values=0)
        # 上下拼接帧
        merged_frame = np.concatenate((frame1, frame2_resized), axis=0)

        # 输出拼接帧
        out.write(merged_frame)
        print('finish merging {}{}th frame'.format(text, iframe + 1))

    # 释放资源
    tif_video.release()
    stim_video.release()
    out.release()


# %%
def plot_heatmap(data, xlable=None, ylable=None, cmap='coolwarm', vmin=None, vmax=None, title=None, outfile=None, dpi=300, annot=True):
    import seaborn as sns
    import matplotlib.pyplot as plt

    if vmin == None:
        vmin = data.min()
    if vmax == None:
        vmax = data.max()

    fig, ax = plt.subplots(figsize=(data.shape[1], data.shape[0]))
    fig.set_facecolor('white')
    # 使用seaborn的heatmap函数来绘制热图
    sns.heatmap(data, cmap=cmap, vmin=vmin, vmax=vmax, annot=annot, fmt=".6f", annot_kws={"size": 8, "color": 'black'},
                ax=ax, cbar=True, square=True, linewidths=0)

    if xlable is not None:
        ax.set_xticklabels(xlable, rotation=45, ha='right', fontsize=10)
    if ylable is not None:
        ax.set_yticklabels(ylable, rotation=0, fontsize=10)
    plt.gca().invert_yaxis()  # 倒置y轴

    if title:
        ax.set_title(title, fontsize=15)
    if outfile:
        plt.savefig(outfile, bbox_inches='tight', dpi=dpi)

    plt.show()


# %%
def plot_pca(data_pca, n_patch, n_movie, patch_list, movie_name_list, title='', outpath=None, pre_length=0,
             n_frame=None):
    for i_pc in range(3):  # 画第几主成分

        fig = plt.figure(figsize=(int(n_movie * 13), int(n_patch * 3)))
        gs = fig.add_gridspec(n_patch, n_movie)

        # 添加movie名
        for i_movie in range(n_movie):
            ax_movie = fig.add_subplot(gs[0, i_movie])
            ax_movie.text(0.5, 1, movie_name_list[i_movie], fontsize=20, ha='center', va='bottom')
            ax_movie.axis('off')

        for i_patch in range(n_patch):
            for i_movie in range(n_movie):
                # 添加patch名
                gs_patch = gs[i_patch, i_movie].subgridspec(1, 2, width_ratios=[0.03, 1])
                ax_patch_name = fig.add_subplot(gs_patch[0])
                ax_patch_name.text(0, 0.5, patch_list[i_patch], fontsize=15, ha='center', va='center')
                ax_patch_name.axis('off')

                # 添加数据图
                ax_data = fig.add_subplot(gs_patch[1])
                mean = np.mean(data_pca[i_patch, i_movie, i_pc, :, :], axis=-1)
                ax_data.plot(mean, color='black')
                y1 = mean + np.std(data_pca[i_patch, i_movie, i_pc, :, :], axis=-1)
                y2 = mean - np.std(data_pca[i_patch, i_movie, i_pc, :, :], axis=-1)
                ax_data.fill_between(np.arange(data_pca.shape[3]), y1, y2, alpha=0.3, color='steelblue')
                if n_frame is not None:
                    ax_data.axvspan(pre_length, pre_length + n_frame, color='gray', alpha=0.25)

                # 设置横坐标标签
                xticks = np.arange(0, data_pca.shape[3] + 1, 10)  # 每100个数据标注一次
                ax_data.set_xticks(xticks)
                ax_data.set_xticklabels([f"{(x - pre_length) / 10:.0f}s" for x in xticks])
                ax_data.set_xlim(0, data_pca.shape[3])

        fig.set_facecolor('white')
        fig.suptitle(title + ' PC' + str(i_pc + 1), fontsize=30, y=0.99)
        plt.tight_layout()
        if outpath is not None:
            plt.savefig(pjoin(outpath, 'pc{}.png'.format(i_pc + 1)))
        plt.show()


#%%
def subplot_timecourse(data, patch_list, movie_name_list, title='', outpath=None, plot_rep=False, pre_length=0,
             n_frame=None, dpi=300):
    n_patch = data.shape[0]
    n_movie = data.shape[1]
    fig = plt.figure(figsize=(int(n_movie * 10), int(n_patch * 3)))
    gs = fig.add_gridspec(n_patch, n_movie)

    # 添加movie名
    for i_movie in range(n_movie):
        ax_movie = fig.add_subplot(gs[0, i_movie])
        ax_movie.text(0.5, 1, movie_name_list[i_movie], fontsize=20, ha='center', va='bottom')
        ax_movie.axis('off')

    for i_patch in range(n_patch):
        for i_movie in range(n_movie):
            # 添加patch名
            gs_patch = gs[i_patch, i_movie].subgridspec(1, 2, width_ratios=[0.03, 0.7])
            ax_patch_name = fig.add_subplot(gs_patch[0])
            ax_patch_name.text(0, 0.5, patch_list[i_patch], fontsize=15, ha='center', va='center')
            ax_patch_name.axis('off')

            # 添加数据图
            ax_data = fig.add_subplot(gs_patch[1])
            if plot_rep==True:
                for i_rep in range(data.shape[-1]):
                    ax_data.plot(data[i_patch, i_movie, :, i_rep], color='black')
            else:
                mean = np.mean(data[i_patch, i_movie, :, :], axis=-1)
                ax_data.plot(mean, color='black')
                y1 = mean + np.std(data[i_patch, i_movie, :, :], axis=-1)
                y2 = mean - np.std(data[i_patch, i_movie, :, :], axis=-1)
                ax_data.fill_between(np.arange(data.shape[2]), y1, y2, alpha=0.3, color='steelblue')
            if n_frame is not None:
                ax_data.axvspan(pre_length, pre_length + n_frame, color='gray', alpha=0.25)

            # 设置横坐标标签
            xticks = np.arange(0, data.shape[2] + 1, 10)  # 每100个数据标注一次
            ax_data.set_xticks(xticks)
            ax_data.set_xticklabels([f"{(x - pre_length) / 10:.0f}s" for x in xticks])
            ax_data.set_xlim(0, data.shape[2])

    fig.set_facecolor('white')
    fig.suptitle(title, fontsize=40)
    plt.tight_layout(pad=2.0, h_pad=2, w_pad=3, rect=[0, 0, 1, 0.98])
    if outpath is not None:
        plt.savefig(pjoin(outpath, title+'.png'), dpi=dpi, bbox_inches='tight')
    plt.show()

