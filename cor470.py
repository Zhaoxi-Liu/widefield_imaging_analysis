from wfield import *
from tifffile import imread, imwrite

path = r'C:\data---LZX\24.3.19_C92\retinotopy\process\20240319-203353-retinotopy'
frames_average = path[:-10]+'wfield\\frames_average.npy'

avg = np.load(frames_average)[0]
tif_list = glob(pjoin(path, '*.tif'))
for tif in tif_list:
    img = imread(tif)
    cor470 = img*avg+avg
    imwrite(tif[:-4]+'_cor470.tif', cor470)
    print('finish '+tif[:-4]+'_cor470.tif')

print('\nfinish all')
