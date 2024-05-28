from wfield import *
from tifffile import imread, imwrite

path = r'Y:\WF_VC_liuzhaoxi\24.05.20_H78\natural-movie\process\20240520-180021-natural-movie\20240520-180021-tif'
# frames_average = path[:-10]+'wfield\\frames_average.npy'
frames_average = pjoin(r'Y:\WF_VC_liuzhaoxi\24.05.20_H78\natural-movie\process\20240520-180021-wfield', 'frames_average.npy')

avg = np.load(frames_average)[0]
tif_list = glob(pjoin(path, '*.tif'))
for tif in tif_list:
    img = imread(tif)
    cor470 = (img*avg+avg).astype(np.uint16)
    imwrite(tif[:-4]+'_cor470.tif', cor470, imagej=True)
    print('finish '+tif[:-4]+'_cor470.tif')

print('\nfinish all')


