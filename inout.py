from tqdm import tqdm
from wf_utils import filename2int

import h5py
import numpy as np
import tifffile as tiff


def crop_save(folder, parameters, save_folder=None, n_preview=None, **kwargs):
    '''
    folder: the folder containing the images to crop images and save as tiff,
    there may be more than one folder for each channel
    parameters: the cropping parameters, tuples
    save_folder: the folder to save the cropped images
    '''
    chan1_folder_ls = glob(os.path.join(folder, '*-470'))
    chan2_folder_ls = glob(os.path.join(folder, '*-405'))

    top, left, bottom, right = parameters

    chan1_image_ls = []
    for sub_folder in chan1_folder_ls:
        image_ls = glob(os.path.join(sub_folder, '*.tif'))
        # to make sure the images are in the right order
        image_ls = sorted(image_ls, key=filename2int)
        chan1_image_ls.extend(image_ls)
    chan2_image_ls = []
    for sub_folder in chan2_folder_ls:
        image_ls = glob(os.path.join(sub_folder, '*.tif'))
        image_ls = sorted(image_ls, key=filename2int)
        chan2_image_ls.extend(image_ls)
    # print(len(chan1_image_ls), len(chan2_image_ls))

    n_frames = np.min([len(chan1_image_ls), len(chan2_image_ls)])
    print(n_frames)
    
    if save_folder is None:
        tiff_path = os.path.join(folder, 'merged.tif')
    n_write = n_frames if n_preview is None else n_preview
    with tiff.TiffWriter(tiff_path, bigtiff=True, imagej=True) as tif:
        for i in tqdm(range(n_write)):
            image1 = tiff.imread(chan1_image_ls[i])
            image2 = tiff.imread(chan2_image_ls[i])
            image1_cropped = image1[top:bottom, left:right]
            image2_cropped = image2[top:bottom, left:right]
            _merge = np.stack([image1_cropped, image2_cropped],
                axis=0).astype(np.uint16)
            tif.write(_merge, contiguous=True)

def h5py_write(file_path, data_dict, overwrite=False):
    '''
    data_dict['behavior']['pupil_size'] is a numpy array
    '''
    # if not(os.path.exists(file_path)):
    #     f = h5py.File(file_path, "w")
    #     f.close()
    #     logger.info('h5py_write: The file was not exists, now created'.format(file_path))

    with h5py.File(file_path, "a") as f:
        for grp_name in data_dict:
            # if overwrite:
                # if grp_name in f:
                #     del f[grp_name]
            grp = f.require_group(grp_name)
            for dset_name in data_dict[grp_name]:
                data_name = '{}/{}'.format(grp_name,dset_name)
                # print(data_name)
                if overwrite:
                    if data_name in f:
                        del f[data_name]
                        print('overwrite {} in {}'.format(dset_name, grp_name))
                data = data_dict[grp_name][dset_name]
                if np.issctype(type(data)):
                    data = np.array(data)
                
                dset = grp.require_dataset(dset_name, data.shape, data.dtype)
                dset[()] = data

def h5py_read(file_path,group_read='all',dataset_read='all'):    
    with h5py.File(file_path, "a") as f:
        data_dict = {}
        if group_read == 'all':
            for group in f:
                data_dict[group] = {}
                for dataset in f[group]:
                    data_dict[group][dataset] = f[group][dataset][()]
        elif dataset_read=='all':
            data_dict[group_read] = {}
            for dataset in f[group_read]:
                data_dict[group_read][dataset] = f[group_read][dataset][()]
        else:
            data_dict[group_read] = {}
            data_dict[group_read][dataset_read] = f[group_read][dataset_read][()]
    return data_dict