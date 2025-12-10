import pickle
import h5py
import numpy as np

def save_pkl(filename, save_object):
	writer = open(filename,'wb')
	pickle.dump(save_object, writer)
	writer.close()

def load_pkl(filename):
	loader = open(filename,'rb')
	file = pickle.load(loader)
	loader.close()
	return file


def save_hdf5(output_path, asset_dict, attr_dict= None, mode='a', chunk_size=32):
    """
    Save data to HDF5 file. Handles both numpy arrays and TensorFlow tensors.
    
    Args:
        output_path: Path to output HDF5 file
        asset_dict: Dictionary of data to save (values can be numpy arrays or TF tensors)
        attr_dict: Optional dictionary of attributes to save
        mode: File mode ('w' for write, 'a' for append)
        chunk_size: Chunk size for HDF5 dataset
    """
    with h5py.File(output_path, mode) as file:
        for key, val in asset_dict.items():
            # Convert TensorFlow tensor to numpy array if needed
            if hasattr(val, 'numpy'):
                # TensorFlow tensor
                val = val.numpy()
            elif not isinstance(val, np.ndarray):
                # Try to convert to numpy array
                val = np.array(val)
            
            data_shape = val.shape
            if key not in file:
                data_type = val.dtype
                chunk_shape = (chunk_size, ) + data_shape[1:]
                maxshape = (None, ) + data_shape[1:]
                dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape, dtype=data_type)
                dset[:] = val
                if attr_dict is not None:
                    if key in attr_dict.keys():
                        for attr_key, attr_val in attr_dict[key].items():
                            dset.attrs[attr_key] = attr_val
            else:
                dset = file[key]
                dset.resize(len(dset) + data_shape[0], axis=0)
                dset[-data_shape[0]:] = val
    return output_path