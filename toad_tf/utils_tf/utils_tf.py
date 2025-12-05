import tensorflow as tf
import numpy as np
import pdb
import math
from itertools import islice
import collections

def collate_MIL_mtl_concat(batch):
    img = tf.concat([item[0] for item in batch], axis=0)
    label = tf.constant([item[1] for item in batch], dtype=tf.int64)
    site = tf.constant([item[2] for item in batch], dtype=tf.int64)
    sex = tf.constant([item[3] for item in batch], dtype=tf.int64)
    return img, label, site, sex

def get_simple_loader(dataset, batch_size=1):

    def generator():
        batch = []
        for i in range(len(dataset)):
            batch.append(dataset[i])  # each item is (img, label, site, sex)
            if len(batch) == batch_size:
                yield collate_MIL_mtl_concat(batch)
                batch = []
        # last partial batch
        if batch:
            yield collate_MIL_mtl_concat(batch)

    output_signature = (
        tf.TensorSpec(shape=(None, dataset.feat_dim), dtype=tf.float32),  # img bag
        tf.TensorSpec(shape=(None,), dtype=tf.int64),                     # label vector
        tf.TensorSpec(shape=(None,), dtype=tf.int64),                     # site vector
        tf.TensorSpec(shape=(None,), dtype=tf.int64),                     # sex vector
    )

    ds = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    return ds

def get_split_loader(split_dataset, training=False, testing=False, weighted=False):
    """
    return either the validation loader or training loader 
    """
    N = len(split_dataset)

    if not testing:
        if training:
            if weighted:
                weights = make_weights_for_balanced_classes_split(split_dataset).numpy()
                weights = weights / np.sum(weights)
                indices = np.random.choice(np.arange(N), size=N, replace=True, p=weights)
            else:
                indices = np.random.permutation(N)
        else:
            indices = np.arange(N)
    else:
        ids = np.random.choice(np.arange(N), int(N * 0.01), replace=False)
        indices = np.sort(ids)
        
    def generator():
        for idx in indices:
            img, label, site, sex = split_dataset[idx]
            # mimic PyTorch: batch_size=1, so collate on a singleton list
            yield collate_MIL_mtl_concat([(img, label, site, sex)])
            
    output_signature = (
        tf.TensorSpec(shape=(None, split_dataset.feat_dim), dtype=tf.float32),  # img bag
        tf.TensorSpec(shape=(1,), dtype=tf.int64),                               # label
        tf.TensorSpec(shape=(1,), dtype=tf.int64),                               # site
        tf.TensorSpec(shape=(1,), dtype=tf.int64),                               # sex
    )

    ds = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    return ds

def get_optim(model, args):
    if args.opt == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr, weight_decay=args.reg)
    elif args.opt == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=args.lr, momentum=0.9, weight_decay=args.reg)
    else:
        raise NotImplementedError
    return optimizer

def print_network(net):
	if not net.built:
		return
	num_params = sum(tf.size(v).numpy() for v in net.variables)
	num_params_train = sum(tf.size(v).numpy() for v in net.trainable_variables)
	print(net)
	print("Total parameters:", num_params)
	print("Trainable parameters:", num_params_train)

def generate_split(cls_ids, val_num, test_num, samples, n_splits = 5,
	seed = 7, label_frac = 1.0, custom_test_ids = None):
	indices = np.arange(samples).astype(int)

	if custom_test_ids is not None:
		indices = np.setdiff1d(indices, custom_test_ids)

	np.random.seed(seed)
	for i in range(n_splits):
		all_val_ids = []
		all_test_ids = []
		sampled_train_ids = []
		
		if custom_test_ids is not None: # pre-built test split, do not need to sample
			all_test_ids.extend(custom_test_ids)

		for c in range(len(val_num)):
			possible_indices = np.intersect1d(cls_ids[c], indices) #all indices of this class
			remaining_ids = possible_indices

			if val_num[c] > 0:
				val_ids = np.random.choice(possible_indices, val_num[c], replace = False) # validation ids
				remaining_ids = np.setdiff1d(possible_indices, val_ids) #indices of this class left after validation
				all_val_ids.extend(val_ids)

			if custom_test_ids is None and test_num[c] > 0: # sample test split

				test_ids = np.random.choice(remaining_ids, test_num[c], replace = False)
				remaining_ids = np.setdiff1d(remaining_ids, test_ids)
				all_test_ids.extend(test_ids)

			if label_frac == 1:
				sampled_train_ids.extend(remaining_ids)
			
			else:
				sample_num  = math.ceil(len(remaining_ids) * label_frac)
				slice_ids = np.arange(sample_num)
				sampled_train_ids.extend(remaining_ids[slice_ids])

		yield sampled_train_ids, all_val_ids, all_test_ids


def nth(iterator, n, default=None):
	if n is None:
		return collections.deque(iterator, maxlen=0)
	else:
		return next(islice(iterator,n, None), default)

def calculate_error(Y_hat, Y):
	error = 1.0 - float(tf.reduce_mean(tf.cast(tf.equal(Y_hat, Y), tf.float32)).numpy())
	return error

def make_weights_for_balanced_classes_split(dataset):
    N = float(len(dataset))                                           
    weight_per_class = [N/len(dataset.slide_cls_ids[c]) for c in range(len(dataset.slide_cls_ids))]                                                                                                     
    weight = [0] * int(N)                                           
    for idx in range(len(dataset)):   
        y = dataset.getlabel(idx)                        
        weight[idx] = weight_per_class[y]                                  

    return tf.constant(weight, dtype=tf.float64)

def initialize_weights(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.kernel.assign(tf.keras.initializers.GlorotNormal()(shape=layer.kernel.shape))
            if layer.bias is not None:
                layer.bias.assign(tf.zeros(layer.bias.shape))
