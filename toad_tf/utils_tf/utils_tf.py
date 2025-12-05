import tensorflow as tf
from torchvision import transforms
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler, sampler
import torch.nn.functional as F

import numpy as np
import pdb
import math
from itertools import islice
import collections

class SubsetSequentialSampler(Sampler):
	"""Samples elements sequentially from a given list of indices, without replacement.

	Arguments:
		indices (sequence): a sequence of indices
	"""
	def __init__(self, indices):
		self.indices = indices

	def __iter__(self):
		return iter(self.indices)

	def __len__(self):
		return len(self.indices)

def collate_MIL_mtl_concat(batch):
    img = tf.concat([item[0] for item in batch], axis=0)
    label = tf.constant([item[1] for item in batch], dtype=tf.int64)
    site = tf.constant([item[2] for item in batch], dtype=tf.int64)
    sex = tf.constant([item[3] for item in batch], dtype=tf.int64)
    return [img, label, site, sex]

def get_simple_loader(dataset, batch_size=1):
	kwargs = {'num_workers': 4} if device.type == "cuda" else {}
	loader = DataLoader(dataset, batch_size=batch_size, sampler = sampler.SequentialSampler(dataset), collate_fn = collate_MIL_mtl_concat, **kwargs)
	return loader 

def get_split_loader(split_dataset, training = False, testing = False, weighted = False):
	"""
		return either the validation loader or training loader 
	"""
	kwargs = {'num_workers': 4} if device.type == "cuda" else {}
	if not testing:
		if training:
			if weighted:
				weights = make_weights_for_balanced_classes_split(split_dataset)
				loader = DataLoader(split_dataset, batch_size=1, sampler = WeightedRandomSampler(weights, len(weights)), collate_fn = collate_MIL_mtl_concat, **kwargs)	
			else:
				loader = DataLoader(split_dataset, batch_size=1, sampler = RandomSampler(split_dataset), collate_fn = collate_MIL_mtl_concat, **kwargs)
		else:
			loader = DataLoader(split_dataset, batch_size=1, sampler = SequentialSampler(split_dataset), collate_fn = collate_MIL_mtl_concat, **kwargs)
	
	else:
		ids = np.random.choice(np.arange(len(split_dataset)), int(len(split_dataset)*0.01), replace = False)
		loader = DataLoader(split_dataset, batch_size=1, sampler = SubsetSequentialSampler(ids), collate_fn = collate_MIL_mtl_concat, **kwargs )

	return loader

def get_optim(model, args):
    if args.opt == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr, weight_decay=args.reg)
    elif args.opt == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=args.lr, momentum=0.9, weight_decay=args.reg)
    else:
        raise NotImplementedError
    return optimizer

def print_network(net):
	num_params = 0
	num_params_train = 0
	print(net)
	
	for param in net.parameters():
		n = param.numel()
		num_params += n
		if param.requires_grad:
			num_params_train += n
	
	print('Total number of parameters: %d' % num_params)
	print('Total number of trainable parameters: %d' % num_params_train)


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
	error = 1. - Y_hat.float().eq(Y.float()).float().mean().item()

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
            layer.bias.assign(tf.zeros(layer.bias.shape))
