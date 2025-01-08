from wfield import *
from tifffile import imread, imwrite


#%%
def rolling_mean_of_image_ls(image_array, window_size=5):
    padded_image_array = []
    shape = list(image_array.shape)
    shape[0] += window_size
    for i in range(window_size):
        tmp = np.zeros(shape)
        tmp.fill(np.nan)
        tmp[i:i - window_size] = image_array
        padded_image_array.append(tmp)

    mean = np.nanmean(padded_image_array, axis=0)

    return mean[window_size // 2:window_size // 2 - window_size]


#%%
def enhance_470(img470, mean_image):
    average_ls = []
    dx_ls = []
    for i in range(img470.shape[0]):
        dx = img470[i, :, :] - mean_image
        dx_ls.append(dx)
    dx_array = np.array(dx_ls)

    # Perform rolling mean on each array in the list
    rolling_means = rolling_mean_of_image_ls(dx_array, window_size = 5)
    for i in range(img470.shape[0]):
        dx = rolling_means[i]
        average_ls.append(img470[i, :, :] + 10 * dx)
    image_stack = np.array(average_ls)
    return image_stack

#%%
def enhance_df_f(df_f, mean_image):
    df = df_f * mean_image

    # Perform rolling mean on each array in the list
    rolling_means = rolling_mean_of_image_ls(df, window_size = 5)
    average_ls = []
    # for i in range(df_f.shape[0]):
    #     dx = rolling_means[i]
    #     average_ls.append(df_f[i, :, :] + 10 * dx)
    # image_stack = np.array(average_ls)
    image_stack = df_f * mean_image + mean_image + 10 * rolling_means
    return image_stack


'''
#%%
path_wfield = r'Y:\WF_VC_liuzhaoxi\24.06.20_H78\moving-bar\process\20240620-172655-wfield'
experiment = os.path.basename(path_wfield)[:15]
path_out = pjoin(path_wfield, '..', experiment + '-natural-movie')
frames_average = pjoin(r'Y:\WF_VC_liuzhaoxi\24.06.20_H78\moving-bar\process\20240620-172655-wfield', 'frames_average.npy')
avg = np.load(frames_average)[0]


#%% cor470
tif_list = glob(pjoin(path, 'tif/avg*.tif'))
for tif in tif_list:
    img = imread(tif)
    cor470 = (img*avg+avg).astype(np.uint16)
    imwrite(tif[:-4]+'_cor470.tif', cor470, imagej=True)
    print('finish '+tif[:-4]+'_cor470.tif')

print('\nfinish all')


#%% enhance
# tif_list = glob(pjoin(path, r'tif-cor470\*cor470.tif'))
tif_list = glob(pjoin(path_out, r'20240520-180021-tif-rep\*rep.tif'))
for tif in tif_list:
    print('start '+tif)
    img = imread(tif)
    img_enhance = enhance_df_f(img, avg)
    imwrite(tif[:-4]+'-enhance.tif', img_enhance.astype('uint16'), imagej=True)
    print('finish'+tif[:-4]+'-enhance.tif')


#%% reshape
tif_list = glob(pjoin(path_out, r'20240520-180021-tif-rep-enhance\rep*.tif'))
for tif_file in tif_list:
    tif = imread(tif_file)
    tif_reshape = tif.reshape(10,-1,*tif.shape[1:])
    tif_reshape1 = np.concatenate(np.split(tif_reshape, 5, axis=0), axis=-1).squeeze()
    # tif_reshape2 = np.concatenate(np.split(tif_reshape1, 2, axis=0), axis=-2).squeeze()
    imwrite(tif_file[:-4]+'-reshape.tif', tif_reshape1, imagej=True)
    print('finish reshape '+os.path.basename(tif_file))


'''