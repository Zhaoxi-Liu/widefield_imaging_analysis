
def denoise(y):
    import numpy as np
    # 计算观测值与前后数据点的差
    diff_y_prev = np.diff(y, prepend=np.nan)
    diff_y_next = np.diff(y, append=np.nan)
    # 计算标准差的3倍
    three_times_std = 3 * np.std(np.diff(y))
    # 找到差值都大于3倍标准差的数据点的索引
    outliers_indices = np.where((np.abs(diff_y_prev) > three_times_std) & (np.abs(diff_y_next) > three_times_std))[0]
    # 用相邻两个值的平均数替代异常值
    for i in outliers_indices:
        y[i] = (y[i-1] + y[i+1]) / 2

    return y


def plot_outliers(x, y, ax, label):
    import numpy as np
    import matplotlib.pyplot as plt
    # 分别计算观测值与前后数据点的差
    diff_y_prev = np.concatenate(([np.nan], np.diff(y)))
    diff_y_next = np.concatenate((np.diff(y), [np.nan]))
    # 计算标准差的3倍
    three_times_std = 3 * np.std(np.diff(y))
    # 找到前后差值都大于3倍标准差的数据点
    outliers_indices = np.where((np.abs(diff_y_prev) > three_times_std) & (np.abs(diff_y_next) > three_times_std))
    # 在折线图上标记异常点
    ax.plot(x[outliers_indices], y[outliers_indices], 'ko', label=label + ' Outliers', fillstyle='none', markerfacecolor='none')


def plotFluor(path, trial):
    # trial 格式类似于 "20230904-225959"
    # path 是刺激文件夹，包含raw和process子文件夹

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    file_405 = trial + "-405-Values.csv"
    file_470 = trial + "-470-Values.csv"
    # 使用 pandas 读取 CSV 文件内容
    dat_405 = pd.read_csv(os.path.join(path, "process", file_405), header=None).values.flatten()
    dat_470 = pd.read_csv(os.path.join(path, "process", file_470), header=None).values.flatten()

    # 提取荧光值
    # 两通道可能差一张图
    n_frame = min(dat_405.shape[0], dat_470.shape[0])
    y_405 = dat_405[:n_frame]
    y_470 = dat_470[:n_frame]

    # 去掉一些串道之类的错误点
    # y_405 = denoise(y_405)
    # y_470 = denoise(y_470)

    # 计算470和405的差值
    y_470_405 = y_470 - y_405
    y_470_405_scaled = y_470_405 - np.mean(y_470_405)

    # 创建一个新的图像
    fig, axs = plt.subplots(3, 1, figsize=(12, 6), gridspec_kw={'height_ratios': [1.5, 1, 0.2]})

    # 绘制双通道荧光值
    axs[0].plot(y_405, color='#b71c1c', label='405', linewidth=0.5)
    axs[0].plot(y_470, color='#00796b', label='470', linewidth=0.5)
    # plot_outliers(np.arange(n_frame), y_405, axs[0], '405')
    # plot_outliers(np.arange(n_frame), y_470, axs[0], '470')
    axs[0].set_ylabel('fluor value', fontsize=12)
    axs[0].set_title('2 channel', fontsize=12)
    axs[0].legend()
    axs[0].grid(True)
    axs[0].set_xlim([0, n_frame])

    # 绘制470-405的荧光值
    axs[1].plot(y_470_405_scaled, color='#0097a7', linewidth=0.5)
    # plot_outliers(np.arange(n_frame), y_470_405_scaled, axs[1], '470-405')
    axs[1].set_ylabel('Δ fluor value', fontsize=12)
    axs[1].set_title('scaled 470-405', fontsize=12)
    axs[1].grid(True)
    axs[1].set_xlim([0, n_frame])

    # 读取刺激时间
    stimfile = pd.read_csv(os.path.join(path, "raw", trial + ".csv"), header=None).values
    stim_delay = pd.read_csv(os.path.join(path, "raw", trial + "-470Timestamp.csv"), header=None).values
    stim_delay = int(stim_delay[0] / 10)
    stim = np.zeros(n_frame)
    for i in range(n_frame):
        stim[i] = stimfile[(i * 10 + stim_delay), 0]

    # 绘制刺激时间
    axs[2].fill_between(range(n_frame), stim, color='k')
    axs[2].set_xlim([0, n_frame])
    axs[2].axis('off')
    axs[2].annotate('Stimuli', xy=(-0.1, 0.5), xycoords='axes fraction',
                    fontsize=12, rotation=0, va='center')

    # 添加总标题
    plt.suptitle(trial, fontsize=16)
    plt.subplots_adjust(hspace=0.6)  # 调整子图之间的垂直间隔

    # 保存图像为PNG格式
    outputFileName = trial + "_value_stimu.png"
    plt.savefig(os.path.join(path, "process", outputFileName), dpi=300, bbox_inches='tight')
    plt.show()


