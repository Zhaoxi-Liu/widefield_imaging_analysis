import pandas as pd
from os.path import join as pjoin
import os
import cv2
import numpy as np
import moten
import matplotlib.pyplot as plt


# %%
def process_video(video_path, target_sec=None, downsample_factor=10):
    """
    处理单个视频，提取亮度、对比度、运动能量和帧差特征。

    参数:
        video_path (str): 视频文件路径
        target_sec (float): 读取多少秒（None表示全部读取）
        downsample_factor (int): 空间降采样因子

    返回:
        dict: 包含亮度、对比度、运动能量、帧差和FPS的字典
    """
    # 读取视频
    cap = cv2.VideoCapture(video_path)
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 计算目标帧数
    if target_sec is not None:
        target_frames = min(total_frames, int(target_sec * fps))
    else:
        target_frames = total_frames

    # 一次性加载所有帧并降采样
    frames = []
    for _ in range(target_frames):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 抗锯齿降采样
        h, w = gray.shape
        new_h, new_w = h // downsample_factor, w // downsample_factor
        resized = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        frames.append(resized)
    cap.release()

    if not frames:
        raise ValueError(f"视频 {video_path} 无有效帧或无法读取")

    frames = np.stack(frames)  # 形状: [T, H, W]
    T, H, W = frames.shape

    # 计算亮度（均值）和对比度（标准差）
    brightness = np.mean(frames, axis=(1, 2))  # [T,]
    contrast = np.std(frames, axis=(1, 2))  # [T,]

    # 计算运动能量（使用moten）
    pyramid = moten.get_default_pyramid(vhsize=(H, W), fps=fps)
    moten_features = pyramid.project_stimulus(frames)  # [T, n_filters]
    motion_energy = np.mean(moten_features, axis=1)  # [T,]

    # 计算帧差（|后一帧 - 前一帧|的均值）,末尾补0
    frame_diff = np.mean(np.abs(np.diff(frames, axis=0)), axis=(1, 2))  # [T-1,]
    frame_diff = np.pad(frame_diff, pad_width=(0, 1), mode='constant', constant_values=0)

    # 与钙成像频率对齐(fpi: frame per image)
    fpi = round(fps/10)
    return {
        'brightness': brightness.reshape(-1,fpi).mean(axis=1),
        'contrast': contrast.reshape(-1,fpi).mean(axis=1),
        'motion_energy': motion_energy.reshape(-1,fpi).mean(axis=1),
        'frame_diff': frame_diff.reshape(-1,fpi).mean(axis=1),
        'fps': fps
    }


# %%

def process_all_videos(video_folder, video_list, target_sec=None, downsample_factor=10):
    """
    处理所有视频并合并结果

    参数:
        video_folder (str): 视频文件夹路径
        video_list (list): 视频文件名列表
        target_sec (float): 读取多少秒（None表示全部读取）
        downsample_factor (int): 空间降采样因子

    返回:
        dict: 包含四个特征矩阵的字典
    """
    brightness_matrix = []
    contrast_matrix = []
    motion_energy_matrix = []
    frame_diff_matrix = []
    fps_list = []

    for video_name in video_list:
        video_path = os.path.join(video_folder, str(video_name))
        print(f"正在处理视频: {video_name}")

        result = process_video(video_path, target_sec, downsample_factor)
        brightness_matrix.append(result['brightness'])
        contrast_matrix.append(result['contrast'])
        motion_energy_matrix.append(result['motion_energy'])
        frame_diff_matrix.append(result['frame_diff'])
        fps_list.append(result['fps'])

    # 对齐所有视频的帧数（以最短视频为准）
    min_frames = min(len(b) for b in brightness_matrix)
    brightness_matrix = np.array([x[:min_frames] for x in brightness_matrix])
    contrast_matrix = np.array([x[:min_frames] for x in contrast_matrix])
    motion_energy_matrix = np.array([x[:min_frames] for x in motion_energy_matrix])
    frame_diff_matrix = np.array([x[:min_frames] for x in frame_diff_matrix])

    return {
        'brightness': brightness_matrix,  # 形状: [n_videos, T]
        'contrast': contrast_matrix,  # 形状: [n_videos, T]
        'motion_energy': motion_energy_matrix,  # 形状: [n_videos, T]
        'frame_diff': frame_diff_matrix,  # 形状: [n_videos, T-1]
        'fps': fps_list
    }


# %%
movie_folder = pjoin(r'D:\Zhaoxi\mouse_vision\Matlab-code\LZX_stimuli\yatang_phychtoolstimgui_LCr', 'natural_movies')
movie_list = pd.read_csv(pjoin(movie_folder, 'movie_list.txt'), header=None).values
n_movie = movie_list.size
movie_name_list = []
for imovie in range(n_movie):
    movie_name_list.append(str(movie_list[imovie])[2:-2])
#%%
target_sec = 15  # 每个视频读取多少秒（None表示全部读取）
downsample_factor = 10  # 空间降采样因子

# 处理所有视频
results = process_all_videos(movie_folder, movie_name_list, target_sec, downsample_factor)

# 打印结果形状
print(f"亮度矩阵形状: {results['brightness'].shape}")
print(f"对比度矩阵形状: {results['contrast'].shape}")
print(f"运动能量矩阵形状: {results['motion_energy'].shape}")
print(f"帧差矩阵形状: {results['frame_diff'].shape}")
print(f"视频FPS列表: {results['fps']}")
# %%
# 保存结果
np.savez("video_features.npz",
         brightness=results['brightness'],
         contrast=results['contrast'],
         motion_energy=results['motion_energy'],
         frame_diff=results['frame_diff'],
         fps=results['fps'])

# %%
with np.load("video_features.npz") as data:
    brightness, contrast, motion_energy, frame_diff, fps = \
        data['brightness'], data['contrast'], data['motion_energy'], data['frame_diff'], data['fps']


# %%
from scipy.stats import zscore

fig, axs = plt.subplots(nrows=14, ncols=1, figsize=(12, 30), sharex=False)
plt.subplots_adjust(hspace=0.5)  # 调整子图间距

# 为每个视频绘制曲线
for i in range(n_movie):
    ax = axs[i]

    # 获取当前视频的数据
    movie_brightness = zscore(brightness[i])
    movie_contrast = zscore(contrast[i])
    movie_motion_energy = zscore(motion_energy[i])
    movie_frame_diff = zscore(frame_diff[i])

    # 绘制曲线
    frames = np.arange(brightness.shape[1])
    ax.plot(frames, movie_brightness, label='Brightness', color='blue', alpha=0.7)
    ax.plot(frames, movie_contrast, label='Contrast', color='green', alpha=0.7)
    ax.plot(frames, movie_motion_energy, label='Motion Energy', color='red', alpha=0.7)
    ax.plot(frames, movie_frame_diff, label='Frame Diff', color='purple', alpha=0.7)

    # 添加标题和网格
    ax.set_title(f'{movie_name_list[i][:-4]}', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)

    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
    ax.set_xlim(0, brightness.shape[1])
    ax.set_xticks(np.arange(0, brightness.shape[1]+1, 10))


# 添加共享的x轴和y轴标签
fig.text(0.5, 0.01, 'Frame Number', ha='center', va='center', fontsize=12)
fig.text(0.015, 0.5, 'z-score Value', ha='center', va='center', rotation='vertical', fontsize=12)

plt.suptitle('Video Features Analysis', y=0.99, fontsize=14)
plt.tight_layout(rect=[0.01, 0.02, 1, 0.98])  # 预留顶部和底部空间
plt.savefig('Movie Features Analysis_adjust')
plt.show()

#%%
movie_folder = pjoin(r'D:\Zhaoxi\mouse_vision\Matlab-code\LZX_stimuli\yatang_phychtoolstimgui_LCr', 'natural_movies')
movie_list_adjust = pd.read_csv(pjoin(movie_folder, 'movie_list_adjust.txt'), header=None).values
n_movie_adjust = movie_list_adjust.size
movie_name_list_adjust = []
for imovie in range(n_movie_adjust):
    movie_name_list_adjust.append(str(movie_list_adjust[imovie])[2:-2])

target_sec = 15  # 每个视频读取多少秒（None表示全部读取）
downsample_factor = 10  # 空间降采样因子

# 处理所有视频
results_adjust = process_all_videos(movie_folder, movie_name_list_adjust, target_sec, downsample_factor)

# 打印结果形状
print(f"亮度矩阵形状: {results_adjust['brightness'].shape}")
print(f"对比度矩阵形状: {results_adjust['contrast'].shape}")
print(f"运动能量矩阵形状: {results_adjust['motion_energy'].shape}")
print(f"帧差矩阵形状: {results_adjust['frame_diff'].shape}")
print(f"视频FPS列表: {results_adjust['fps']}")

# 保存结果
np.savez("video_features_adjust.npz",
         brightness=results_adjust['brightness'],
         contrast=results_adjust['contrast'],
         motion_energy=results_adjust['motion_energy'],
         frame_diff=results_adjust['frame_diff'],
         fps=results_adjust['fps'])

#%%
with np.load("video_features_adjust.npz") as data:
    brightness, contrast, motion_energy, frame_diff, fps = \
        data['brightness'], data['contrast'], data['motion_energy'], data['frame_diff'], data['fps']

# %%
area_timecourse = np.load(r'Y:\WF_VC_liuzhaoxi\______________natural_movie_summary2\natural-movie-series\all_area_timecourse.npy')
area_timecourse_mean1 = area_timecourse[:17, 2:, 20:-30,:,:].mean((0, -1))
area_timecourse_mean2 = area_timecourse[17:, 2:, 20:-30,:,:].mean((0, -1))

#%%
area_list = ['MOp','MOs','SSp-n','SSp-m','SSp-un','PL','ACAd','RSPv','RSPd','RSPagl','VISC',
             'SSs','SSp-bfd','SSp-tr','SSp-ll','SSp-ul','TEa','AUDd','AUDp','AUDpo','AUDv',
             'VISli','VISpor','VISpl','VISpm','VISl','VISal','VISrl','VISa','VISam','VISp']
#%%
target_sec = 15  # 每个视频读取多少秒（None表示全部读取）
downsample_factor = 10  # 空间降采样因子

#%%
from scipy.signal import correlate
from scipy.stats import zscore
# 标准化钙信号和刺激参数
# timecourse_z = zscore(area_timecourse, axis=1)
# brightness_z = zscore(brightness, axis=1)
# contrast_z = zscore(contrast, axis=1)
# motion_energy_z = zscore(motion_energy, axis=1)
# frame_diff_z = zscore(frame_diff, axis=1)
timecourse_z = zscore(area_timecourse_mean2.transpose(0,2,1).reshape(-1, n_movie*target_sec*10), axis=1)
brightness_z = zscore(brightness).reshape(-1)
contrast_z = zscore(contrast).reshape(-1)
motion_energy_z = zscore(motion_energy).reshape(-1)
frame_diff_z = zscore(frame_diff).reshape(-1)
#%%
def lagged_corr(x, y, min_lag, max_lag):
    """计算时滞互相关（标准化Pearson）"""
    corr = correlate(x, y, mode='full', method='auto') / (len(x) * np.std(x) * np.std(y))
    mid = len(corr) // 2
    return corr[mid + min_lag : mid + max_lag + 1]

def lagged_corr_feature(feature_z):
    # corr_values = np.zeros((area_timecourse.shape[1],max_lag-min_lag+1))
    # for i_area in range(area_timecourse.shape[1]):
    corr_values = np.zeros((area_timecourse_mean.shape[0], max_lag - min_lag + 1))
    for i_area in range(area_timecourse_mean.shape[0]):
        corr_values[i_area] = lagged_corr(timecourse_z[i_area, :], feature_z, min_lag, max_lag)
    return corr_values

#%%
min_lag, max_lag = -5, 15  # 最大滞后帧数
lags = np.arange(min_lag, max_lag + 1)

corr_values_brightness = lagged_corr_feature(brightness_z)
corr_values_contrast = lagged_corr_feature(contrast_z)
corr_values_motion_energy = lagged_corr_feature(motion_energy_z)
corr_values_frame_diff = lagged_corr_feature(frame_diff_z)

#%%
def plot_heatmap(data, xlabel=None, ylabel=None, xtitle=None, ytitle=None, center=None, cmap='coolwarm', vmin=None, vmax=None, title=None, outfile=None, dpi=300, annot=True):
    import seaborn as sns
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(data.shape[1], data.shape[0]))
    fig.set_facecolor('white')
    # 使用seaborn的heatmap函数来绘制热图
    sns.heatmap(data, cmap=cmap, vmin=vmin, vmax=vmax, center = center, annot=annot, fmt=".3f", annot_kws={"size": 16, "color": 'black'},
                ax=ax, cbar=True, square=True, linewidths=0, cbar_kws={'fraction': 0.03, 'pad': 0.03})
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=35) # 设置colorbar刻度标签的字号

    if xtitle is not None:
        ax.set_xlabel(xtitle, fontsize=50)
    if xlabel is not None:
        ax.set_xticklabels(xlabel, rotation=45, ha='right', fontsize=45)
    if ytitle is not None:
        ax.set_ylabel(ytitle, fontsize=50)
    if ylabel is not None:
        ax.set_yticklabels(ylabel, rotation=0, fontsize=45)
    plt.gca().invert_yaxis()  # 倒置y轴

    plt.tight_layout(rect=[0.01, 0.02, 1, 0.98])  # 预留顶部和底部空间

    if title:
        ax.set_title(title, fontsize=80)
    if outfile:
        plt.savefig(outfile, bbox_inches='tight', dpi=dpi)

    plt.show()

#%%

plot_heatmap(corr_values_contrast, xlabel=lags, ylabel=area_list, xtitle='lags', ytitle='brain areas', cmap='coolwarm',
             vmin=None, vmax=None, center = 0, title='corr_contrast', outfile='corr_contrast.png', dpi=200, annot=False)
plot_heatmap(corr_values_brightness, xlabel=lags, ylabel=area_list, xtitle='lags', ytitle='brain areas',
             cmap='coolwarm', vmin=None, center = 0, vmax=None, title='corr_brightness', outfile='corr_brightness.png', dpi=200,
             annot=False)
plot_heatmap(corr_values_frame_diff, xlabel=lags, ylabel=area_list, xtitle='lags', ytitle='brain areas',
             cmap='coolwarm', vmin=None, vmax=None, center = 0, title='corr_frame_diff', outfile='corr_frame_diff.png', dpi=200,
             annot=False)
plot_heatmap(corr_values_motion_energy, xlabel=lags, ylabel=area_list, xtitle='lags', ytitle='brain areas',
             cmap='coolwarm', vmin=None, vmax=None, center = 0, title='corr_motion_energy', outfile='corr_motion_energy.png',
             dpi=200, annot=False)

# %%
# 画出corr和lag关系
def plot_lag_corr(lags, corr_values, area_list, title=None, outfile=None):
    # 找到最大相关性的时滞
    # peak_lag = lags[np.argmax(np.abs(corr_values))]  # 取绝对值后找峰值
    # plt.axvline(peak_lag, color='r', linestyle='--', label='peak corr lag')
    for i_area in range(len(area_list)):
        plt.plot(lags, corr_values[i_area], label=area_list[i_area])
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
    plt.xlabel('Time Lag (frames)')
    plt.ylabel('Pearson R')
    plt.xlim(min_lag,max_lag)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()  # 预留顶部和底部空间
    if outfile:
        plt.savefig(outfile, bbox_inches='tight')
    plt.show()
#%%
plot_lag_corr(lags, corr_values_brightness[-10:], area_list=area_list[-10:], title='Cross-Correlation with brightness', outfile='Cross-Correlation with brightness1.png')
plot_lag_corr(lags, corr_values_contrast[-10:], area_list=area_list[-10:], title='Cross-Correlation with contrast', outfile='Cross-Correlation with contrast1.png')
plot_lag_corr(lags, corr_values_frame_diff[-10:], area_list=area_list[-10:], title='Cross-Correlation with frame_diff', outfile='Cross-Correlation with frame_diff1.png')
plot_lag_corr(lags, corr_values_motion_energy[-10:], area_list=area_list[-10:], title='Cross-Correlation with motion_energy', outfile='Cross-Correlation with motion_energy1.png')

# %%
def plot_heatmap_4in1(data_list, data_titles, lags, area_list, xtitle='lags', ytitle='brain areas',
                      cmap='coolwarm', center=0, outfile=None, dpi=300):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np

    all_data = np.concatenate([d.ravel() for d in data_list])
    vmax = np.nanmax(all_data)
    vmin = np.nanmin(all_data)
    # -------- figure --------
    fig, axes = plt.subplots(1, 4, figsize=(4 * len(lags)//1.5, len(area_list)), sharey=True)
    fig.subplots_adjust(wspace=0.2)
    fig.set_facecolor('white')

    # x 轴刻度：lags / 10，保留 1 位小数
    xlabels = [f'{x / 10:.1f}' if i % 5 == 0 else ''
        for i, x in enumerate(lags)]

    mappable = None

    for ax, data, title in zip(axes, data_list, data_titles):
        hm = sns.heatmap(
            data, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, center=center,
            square=False, linewidths=0, cbar=False,
            xticklabels=xlabels, yticklabels=area_list
        )

        ax.set_title(title, fontsize=60, pad=40)
        ax.tick_params(axis='x', labelsize=40)
        ax.tick_params(axis='y', labelsize=40, labelrotation=0)
        ax.invert_yaxis()

        if mappable is None:
            mappable = hm.collections[0]

    axes[0].set_ylabel(ytitle, fontsize=50, labelpad=20)
    fig.supxlabel('time lag (second)', fontsize=50, y=0.05)

    # -------- 统一 colorbar --------
    cbar = fig.colorbar(mappable, ax=axes, pad=0.02, shrink=0.8
                        # , fraction=0.03, aspect=30
                        )
    cbar.ax.tick_params(labelsize=35)
    cbar.set_label('Pearson R', fontsize=50, labelpad=15)

    if outfile:
        plt.savefig(outfile, dpi=dpi, bbox_inches='tight')

    plt.show()


#%%
plot_heatmap_4in1(
    data_list=[
        corr_values_brightness,
        corr_values_contrast,
        corr_values_frame_diff,
        corr_values_motion_energy
    ],
    data_titles=['brightness', 'contrast', 'frame diff', 'motion energy'],
    lags=lags, area_list=area_list, center=0, dpi=200,
    outfile='corr_all_4in1.png',
)

#%%
def plot_lag_corr_4in1(lags, corr_values_list, area_list, titles, outfile=None):

    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter

    fig, axes = plt.subplots(1, 4, figsize=(16, 6), sharey=False)
    fig.subplots_adjust(wspace=0.3)
    fig.set_facecolor('white')

    for ax, corr_values, title in zip(axes, corr_values_list, titles):
        for i_area in range(len(area_list)):
            ax.plot(lags, corr_values[i_area], label=area_list[i_area])

        ax.set_title(title, fontsize=20, pad=10)
        ax.set_xlim(min_lag, max_lag)
        ax.grid(True)
        ax.tick_params(axis='both', labelsize=14)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        # 每隔 5 个 lag 一个刻度
        tick_idx = np.arange(0, len(lags), 5)
        tick_lags = np.array(lags)[tick_idx]
        ax.set_xticks(tick_lags)
        ax.set_xticklabels([f"{x * 0.1:.1f}" for x in tick_lags])
    axes[0].set_ylabel('Pearson R', fontsize=16)

    # 统一 X 轴标签
    fig.supxlabel('time lag (second)', fontsize=18, y=0.008)

    # 只放一个 legend（放在最右侧子图外）
    axes[-1].legend(
        loc='upper left',
        bbox_to_anchor=(1.05, 1),
        fontsize=16,
        frameon=False
    )

    if outfile:
        plt.savefig(outfile, bbox_inches='tight', dpi=300)

    plt.show()

#%%
plot_lag_corr_4in1(
    lags=lags,
    corr_values_list=[
        corr_values_brightness[-10:],
        corr_values_contrast[-10:],
        corr_values_frame_diff[-10:],
        corr_values_motion_energy[-10:]
    ],
    area_list=area_list[-10:],
    titles=[
        'brightness',
        'contrast',
        'frame diff',
        'motion energy'
    ],
    outfile='Cross-Correlation_4in1.png'
)

#%%
# area_timecourse = np.load(r'Y:\WF_VC_liuzhaoxi\25.01.10_P41\natural-movie\process\20250110-013702-natural-movie--\P41_20250110-013702_all_area_timecourse.npy')
area_timecourse = np.load(r'Y:\WF_VC_liuzhaoxi\25.01.11_U26\natural-movie\process\20250111-214859-natural-movie--\U26_20250111-214859_all_area_timecourse.npy')
area_timecourse_mean = area_timecourse[2:, 20:-30,:,2]
timecourse_z = zscore(area_timecourse_mean.transpose(0,2,1).reshape(-1, n_movie*target_sec*10), axis=1)
brightness_z = zscore(brightness).reshape(-1)
corr_values_brightness = lagged_corr_feature(brightness_z)
plot_lag_corr(lags, corr_values_brightness[-10:], area_list=area_list[-10:], title='Cross-Correlation with brightness', outfile='Cross-Correlation with brightness1.png')

#%%
np.savez(
    "corr_values.npz",
    brightness=corr_values_brightness,
    contrast=corr_values_contrast,
    frame_diff=corr_values_frame_diff,
    motion_energy=corr_values_motion_energy,
    lags = lags
)
#%%
data = np.load("corr_values.npz")
corr_values_brightness    = data['brightness']
corr_values_contrast      = data['contrast']
corr_values_frame_diff    = data['frame_diff']
corr_values_motion_energy = data['motion_energy']
lags                      = data['lags']


