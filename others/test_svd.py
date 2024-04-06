from wfield import *
from tifffile import imwrite
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# load data
localdisk = r'Y:\WF_VC_liuzhaoxi\test\test-svd'
dat_path = glob(pjoin(localdisk, '*.bin'))[0]

dat = mmap_dat(dat_path)
frames_average = np.load(pjoin(localdisk, 'frames_average.npy'))
# imwrite(pjoin(localdisk,"motion_correct.tif"), dat, imagej=True)

U, SVT = approximate_svd(dat, frames_average)
np.save(pjoin(localdisk, 'U.npy'), U)
np.save(pjoin(localdisk, 'SVT.npy'), SVT)
# 从SVT拆分出 Sigma
Sigma = np.sqrt(np.sum(SVT ** 2, axis=1))
Sigma=Sigma[:100]
s=np.diag(Sigma)
us = U[:,:,:100] @ s

#%%
def plot_k_var(s, outpath = None):
    total_variance = np.sum(s ** 2)  # 计算总方差
    captured_variance = np.cumsum(s ** 2) / total_variance * 100  # 捕获的方差百分比
    # 将捕获的方差绘制成图表
    plt.plot(range(1, len(s) + 1), captured_variance, marker='o', markersize=1, linestyle='-', linewidth=0.5)
    plt.title('Explained Variance vs. Number of Components (k)')
    plt.xlabel('Number of Components (k)')
    plt.ylabel('Captured Variance (%)')
    plt.grid(True)
    if outpath is not None:
        plt.savefig(pjoin(outpath, 'svd_var.png'), bbox_inches='tight', transparent=False, facecolor='white')
    plt.show()


#%%
def pca_X(X,centralize=True):
    """
    SVD approach for principal components analysis
    PV: principal score
    PC: principal components
    lambda: variance explained"""
    [n, p] = np.shape(X)
    X_0 = np.zeros((n,p))
    if centralize:
        for i in range(p):
            X_0[:,i] =  X[:,i]-np.mean(X[:,i])
    else:
        X_0 = X
    [U,S,V] = np.linalg.svd(X_0)
    V = V.T
    lamda = np.square(S)/(n-1)
    lamda = lamda/np.sum(lamda)
    PV = V
    PC = X_0@PV #principal components
    return PV, lamda, PC

#%%
def pca_analysis(X, variance_explained, plot_flag=True):
    if plot_flag:
        imshow_X(X, 'original X')
    n_cells, n_samples = np.shape(X)
    pv, lamda, pc = pca_X(X, centralize=True)
    lamda_cumsum = np.cumsum(lamda)
    if plot_flag:
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(fig_width, fig_height / 3))
        axes[0].plot(lamda[:100], marker='o')
        axes[1].plot(lamda_cumsum[:100], marker='o')
        axes[1].set_ylabel('variance explained')
        axes[1].set_xlabel('number of PCs')

    k = np.where(lamda_cumsum > variance_explained)[0][0] + 1
    pc_k = pc[:, :k]
    pv_k = pv[:, :k]

    lamda_array = np.ones([n_samples, 1]) @ np.reshape(lamda, (1, -1))
    pv_scaled = np.multiply(lamda_array, pv)

    if plot_flag:
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(fig_width, np.min([k / 4, fig_height])))
        sns.heatmap(pv_scaled[:, :k].T, cmap='viridis', xticklabels=False, yticklabels=False, ax=axes[0])
        axes[0].set_title(str(k) + ' principal vectors,' + str(variance_explained * 100) + '% of vraiance explained')
        sns.heatmap(pv_k.T, cmap='viridis', xticklabels=False, yticklabels=False, ax=axes[1])

        # plot principal components
        imshow_X(pc[:, :k], 'principal components', sampling_rate=1)
        # visualize the first 2 PCs
        plt.figure(figsize=(fig_width, fig_height / 3))
        plt.scatter(pc[:, 0], pc[:, 1], s=20, facecolors='none', edgecolors='k', marker='^', alpha=0.3)
        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
    # reconstruct X with PCs that explain variance_explained of variance
    k = np.where(lamda_cumsum > variance_explained)[0][0] + 1
    X_0_new = pc[:, :k] @ pv[:, :k].T
    X_new = np.zeros_like(X)
    for i in range(n_samples):
        X_new[:, i] = X_0_new[:, i] + np.mean(X[:, i])
    if plot_flag:
        imshow_X(X_new, 'reconstructed X with ' + str(k) + ' PCs')
    return k, pc_k, pv_scaled, X_new, lamda_cumsum


#%%
# 设置聚类数目
k = 3
US = us
# 使用 KMeans 进行聚类
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(US)
# 获取聚类结果
labels = kmeans.labels_

# 可视化
plt.scatter(US[:, 0], US[:, 1], c=labels, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='X', s=200, c='red', label='Centroids')
plt.title('KMeans Clustering on Principal Component Scores')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()

#%%
# 将聚类标签映射回原始图像的形状
mapped_labels = labels.reshape(512, 512)

# 创建一个图像，每个类别用不同的颜色表示
colored_image = np.zeros((512, 512) + (3,), dtype=np.uint8)
colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]  # 用红、绿、蓝表示 3 类
for i in range(3):
    colored_image[mapped_labels == i] = colors[i]

# 显示结果
plt.imshow(colored_image)
plt.title('Clustered Image with 3 Colors')
plt.show()
