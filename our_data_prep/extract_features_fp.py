import time
import os
import argparse
import pdb
from functools import partial

import tensorflow as tf
from PIL import Image
import h5py
import openslide
from tqdm import tqdm

import numpy as np

from file_utils import save_hdf5
from dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from model import get_encoder

def compute_w_loader(output_path, dataset, model, verbose = 0):
	"""
	args:
		output_path: directory to save computed features (.h5 file)
		dataset: TensorFlow dataset
		model: TensorFlow model
		verbose: level of feedback
	"""
	if verbose > 0:
		# Note: Can't easily count TF dataset batches without consuming it
		# Will show progress without total count
		print('processing batches...')

	mode = 'w'
	for count, data in enumerate(tqdm(dataset)):
		batch = data['img']
		coords = data['coord']
		
		# Convert coords to numpy if it's a TF tensor
		if hasattr(coords, 'numpy'):
			coords = coords.numpy().astype(np.int32)
		else:
			coords = np.array(coords).astype(np.int32)
		
		# Forward pass through model
		features = model(batch, training=False)
		
		# Convert features to numpy if it's a TF tensor
		if hasattr(features, 'numpy'):
			features = features.numpy().astype(np.float32)
		else:
			features = np.array(features).astype(np.float32)

		asset_dict = {'features': features, 'coords': coords}
		save_hdf5(output_path, asset_dict, attr_dict= None, mode=mode)
		mode = 'a'
	
	return output_path


parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--data_h5_dir', type=str, default=None)
parser.add_argument('--data_slide_dir', type=str, default=None)
parser.add_argument('--slide_ext', type=str, default='.svs')
parser.add_argument('--csv_path', type=str, default=None)
parser.add_argument('--feat_dir', type=str, default=None)
parser.add_argument('--model_name', type=str, default='resnet50_trunc', choices=['resnet50_trunc', 'efficientnet', 'uni_v1', 'conch_v1'])
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--target_patch_size', type=int, default=224)
args = parser.parse_args()


if __name__ == '__main__':
	print('initializing dataset')
	csv_path = args.csv_path
	if csv_path is None:
		raise NotImplementedError

	bags_dataset = Dataset_All_Bags(csv_path)
	
	os.makedirs(args.feat_dir, exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'npz_files'), exist_ok=True)
	
	model, img_transforms = get_encoder(args.model_name, target_img_size=args.target_patch_size)
	total = len(bags_dataset)

	for bag_candidate_idx in tqdm(range(total)):
		slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]
		bag_name = slide_id+'.h5'
		h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)
		slide_file_path = os.path.join(args.data_slide_dir, slide_id+args.slide_ext)
		print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
		print(slide_id)

		output_path = os.path.join(args.feat_dir, 'h5_files', bag_name)
		time_start = time.time()
		wsi = openslide.open_slide(slide_file_path)
		dataset = Whole_Slide_Bag_FP(file_path=h5_file_path, 
							   		 wsi=wsi, 
									 img_transforms=img_transforms)

		# Convert to TensorFlow dataset
		tf_dataset = dataset.to_tf_dataset(batch_size=args.batch_size, shuffle=False)
		output_file_path = compute_w_loader(output_path, dataset=tf_dataset, model=model, verbose=1)

		time_elapsed = time.time() - time_start
		print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))

		with h5py.File(output_file_path, "r") as file:
			features = file['features'][:]
			coords = file['coords'][:]
			print('features size: ', features.shape)
			print('coordinates size: ', coords.shape)

		bag_base, _ = os.path.splitext(bag_name)

		np.savez(
			os.path.join(args.feat_dir, 'npz_files', bag_base + '.npz'),
			features=features,
			coords=coords
		)
