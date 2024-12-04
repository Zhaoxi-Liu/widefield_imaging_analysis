import h5py
import numpy as np

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