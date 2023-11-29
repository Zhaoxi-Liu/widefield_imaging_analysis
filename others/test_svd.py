from wfield import *
from tifffile import imwrite

# load data
localdisk = r'Y:\WF_VC_liuzhaoxi\test\test-svd'
dat_path = glob(pjoin(localdisk, '*.bin'))[0]

dat = mmap_dat(dat_path)
frames_average = np.load(pjoin(localdisk, 'frames_average.npy'))
# imwrite(pjoin(localdisk,"motion_correct.tif"), dat, imagej=True)

U, SVT, s = approximate_svd(dat, frames_average)
np.save(pjoin(localdisk, 'U.npy'), U)
np.save(pjoin(localdisk, 'SVT.npy'), SVT)
np.save(pjoin(localdisk, 's.npy'), s)


def calculate_captured_variance(s):
    total_variance = np.sum(s ** 2)  # 计算总方差
    captured_variance = np.cumsum(s ** 2) / total_variance * 100  # 捕获的方差百分比
    return captured_variance


# s是SVD的S值
s = np.load(pjoin(localdisk, 's.npy'))
captured_variance = calculate_captured_variance(s)
# 将捕获的方差绘制成图表
plt.plot(range(1, len(s) + 1), captured_variance, marker='o', markersize=1, linestyle='-', linewidth=0.5)
plt.title('Captured Variance vs. Number of Components (k)')
plt.xlabel('Number of Components (k)')
plt.ylabel('Captured Variance (%)')
plt.grid(True)
plt.savefig(pjoin(localdisk, 'svd_var.png'), bbox_inches='tight', transparent=False, facecolor='white')
plt.show()
