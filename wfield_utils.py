from wfield import *
from tifffile import imwrite
import pandas as pd


'''
def reconstruct(u,svt,dims = None):
    if issparse(u):
        if dims is None:
            raise ValueError('Supply dims = [H,W] when using sparse arrays')
    else:
        if dims is None:
            dims = u.shape[:2]
    return u.dot(svt).reshape((*dims,-1)).transpose(-1,0,1).squeeze()
'''

def svd2tif(path, name='', uint16=False, corr470=False):
    # 读取SVD文件
    U = np.load(pjoin(path, 'U.npy'))
    SVTcorr = np.load(pjoin(path, 'SVTcorr.npy'))
    frames_average = np.load(pjoin(path, 'frames_average.npy'))

    # 重构矫正后图像
    SVD_corr = reconstruct(U, SVTcorr)
    if uint16 is False:
        imwrite(pjoin(path, name+"SVD_corr.tif"), SVD_corr, imagej=True)
        print(name+'SVD_corr.tif输出完成')
    elif uint16 is True:
        # 矫正后图像转成uint16格式
        SVD_corr_uint16 = ((SVD_corr+1)*32768).astype('uint16')
        imwrite(pjoin(path, name+"SVD_corr_uint16.tif"), SVD_corr_uint16, imagej=True)
        print(name+'SVD_corr_uint16.tif输出完成')
    else:
        print('格式不支持，不输出SVD_corr')

    if corr470 is True:
        # 输出矫正后470通道图像
        tif470_corr = (SVD_corr * frames_average[0, :, :]) + frames_average[0, :, :]
        imwrite(pjoin(path, name+"hemo-corr.tif"), tif470_corr.astype('uint16'), dtype='uint16', imagej=True)
        print(name+'hemo-corr.tif"输出完成')



def phasemap(path_merge, nrepeats=10, post_trial=3, export_ave_tif=True, export_phase=True):

    ### 路径和刺激重复次数。（本代码对应的是每个方向连续重复10次再下一个方向）
    path_merge = path_merge
    nrepeats = nrepeats
    post_trial = post_trial   # 刺激消失后多取几秒
    path_out = pjoin(path_merge, '..', os.path.basename(path_merge)[:16] + 'retinotopy')
    os.makedirs(path_out, exist_ok=True)

    ### load data
    trials = pd.read_csv(pjoin(path_merge, "trials.csv"), header=None, dtype=int).values
    nframes_el = min(trials[:20, 3]) + post_trial*10
    nframes_az = min(trials[20:40, 3]) + post_trial*10
    U = np.load(pjoin(path_merge, 'U.npy'))
    SVTcorr = np.load(pjoin(path_merge, 'SVTcorr.npy'))
    nSVD = SVTcorr.shape[0]


    ### extract trial-data for 4 direction stimuli respectively.
    def _sort_frames(nframes, *ntrials):
        avg = np.empty((nSVD, 0, nframes))
        raw = np.empty((nSVD, 0))
        for i in ntrials:
            avg = np.concatenate((avg, SVTcorr[:, trials[i, 1]:trials[i, 1] + nframes].reshape(nSVD, 1, nframes)),
                                 axis=1)
            raw = np.concatenate((raw, SVTcorr[:, trials[i, 1]:trials[i, 1] + nframes]), axis=1)
        avg = np.mean(avg, axis=1)
        return avg, raw

    avg_up, raw_up = _sort_frames(nframes_el, *range(nrepeats * 0, nrepeats * 1))
    avg_down, raw_down = _sort_frames(nframes_el, *range(nrepeats * 1, nrepeats * 2))
    avg_left, raw_left = _sort_frames(nframes_az, *range(nrepeats * 2, nrepeats * 3))
    avg_right, raw_right = _sort_frames(nframes_az, *range(nrepeats * 3, nrepeats * 4))

    if export_ave_tif is True:
        # export trial-average tif
        imwrite(pjoin(path_out, 'avg_up.tif'), reconstruct(U, avg_up).astype('float32'), imagej=True)
        imwrite(pjoin(path_out, 'avg_down.tif'), reconstruct(U, avg_down).astype('float32'), imagej=True)
        imwrite(pjoin(path_out, 'avg_left.tif'), reconstruct(U, avg_left).astype('float32'), imagej=True)
        imwrite(pjoin(path_out, 'avg_right.tif'), reconstruct(U, avg_right).astype('float32'), imagej=True)
        print('Finish exporting trial-average tif')


    ### computes fft in SVD space

    # from scipy.ndimage.filters import gaussian_filter,median_filter
    # mov = runpar(median_filter, U.transpose((2, 0, 1)), size=5)
    # U = np.stack(mov).transpose((1, 2, 0)).astype(np.float32)
    up = reconstruct(U, fft(raw_up.T, axis=0)[nrepeats])
    down = reconstruct(U, fft(raw_down.T, axis=0)[nrepeats])
    left = reconstruct(U, fft(raw_left.T, axis=0)[nrepeats])
    right = reconstruct(U, fft(raw_right.T, axis=0)[nrepeats])
    phase_el = -1. * (np.angle(up) - np.angle(down)) % (2 * np.pi)
    mag_el = (np.abs(up + down) * 2.)
    phase_az = -1. * (np.angle(left) - np.angle(right)) % (2 * np.pi)
    mag_az = (np.abs(left + right) * 2.)

    fig = plt.figure(figsize=[10, 5])
    fig.add_subplot(1, 2, 1)
    plt.imshow(im_fftphase_hsv([mag_el, phase_el]))
    plt.axis('off')
    fig.add_subplot(1, 2, 2)
    plt.imshow(im_fftphase_hsv([mag_az, phase_az]))
    plt.axis('off')
    fig.set_facecolor('white')
    plt.savefig(pjoin(path_out, 'el_az_unfiltered.png'))

    if export_phase is True:
        ### export phase and magnitude
        np.save(pjoin(path_out, 'phase_el.npy'), phase_el)
        np.save(pjoin(path_out, 'phase_az.npy'), phase_az)
        np.save(pjoin(path_out, 'mag_el.npy'), mag_el)
        np.save(pjoin(path_out, 'mag_az.npy'), mag_az)


    ### plot sign maps
    from scipy.ndimage import median_filter
    fig = plt.figure()
    plt.imshow(median_filter(visual_sign_map(phase_az, phase_el), 33),
               cmap='RdBu_r', clim=[-1, 1])
    plt.colorbar(shrink=0.5)
    plt.axis('off')
    fig.set_facecolor('white')
    plt.savefig(pjoin(path_out, 'phasemap.png'))

