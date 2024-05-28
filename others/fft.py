import numpy as np
import matplotlib.pyplot as plt
from wfield import *

fft_result=fft(raw_up.T, axis=0)[:,0]
sampling_frequency = 10  # Hz
# 计算频率轴
N = len(fft_result)  # fft_result 的长度
freq = np.fft.fftfreq(N, d=1/sampling_frequency)
# 绘制频谱图
plt.figure(figsize=(12, 6))
plt.plot(freq[:N//2], np.abs(fft_result[:N//2]))  # 仅绘制正频率部分
plt.title('Frequency Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.grid(False)
plt.savefig('D:\\Zhaoxi\\mouse_vision\\fft_up.png', bbox_inches='tight', transparent=False, facecolor='white')
