import numpy as np
import pandas as pd
import tensorflow as tf

from PIL import Image
import h5py

class Whole_Slide_Bag:
	def __init__(self,
		file_path,
		img_transforms=None):
		"""
		Args:
			file_path (string): Path to the .h5 file containing patched data.
			roi_transforms (callable, optional): Optional transform to be applied on a sample
		"""
		self.roi_transforms = img_transforms
		self.file_path = file_path

		with h5py.File(self.file_path, "r") as f:
			dset = f['imgs']
			self.length = len(dset)

		self.summary()
			
	def __len__(self):
		return self.length

	def summary(self):
		with h5py.File(self.file_path, "r") as hdf5_file:
			dset = hdf5_file['imgs']
			for name, value in dset.attrs.items():
				print(name, value)

		print('transformations:', self.roi_transforms)

	def __getitem__(self, idx):
		with h5py.File(self.file_path,'r') as hdf5_file:
			img = hdf5_file['imgs'][idx]
			coord = hdf5_file['coords'][idx]
		
		img = Image.fromarray(img)
		img = self.roi_transforms(img)
		return {'img': img, 'coord': coord}
	
	def to_tf_dataset(self, batch_size=256, shuffle=False):
		"""Convert to TensorFlow Dataset"""
		def generator():
			for idx in range(self.length):
				yield self[idx]
		
		# Transform outputs HWC format (H, W, 3), so signature should match
		# After batching it becomes (B, H, W, 3)
		output_signature = {
			'img': tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
			'coord': tf.TensorSpec(shape=(2,), dtype=tf.int32)
		}
		
		dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
		if shuffle:
			dataset = dataset.shuffle(buffer_size=min(1000, self.length))
		dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
		return dataset

class Whole_Slide_Bag_FP:
	def __init__(self,
		file_path,
		wsi,
		img_transforms=None):
		"""
		Args:
			file_path (string): Path to the .h5 file containing patched data.
			img_transforms (callable, optional): Optional transform to be applied on a sample
		"""
		self.wsi = wsi
		self.roi_transforms = img_transforms

		self.file_path = file_path

		with h5py.File(self.file_path, "r") as f:
			dset = f['coords']
			self.patch_level = f['coords'].attrs['patch_level']
			self.patch_size = f['coords'].attrs['patch_size']
			self.length = len(dset)
			
		self.summary()
			
	def __len__(self):
		return self.length

	def summary(self):
		hdf5_file = h5py.File(self.file_path, "r")
		dset = hdf5_file['coords']
		for name, value in dset.attrs.items():
			print(name, value)

		print('\nfeature extraction settings')
		print('transformations: ', self.roi_transforms)

	def __getitem__(self, idx):
		with h5py.File(self.file_path,'r') as hdf5_file:
			coord = hdf5_file['coords'][idx]
		img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')

		img = self.roi_transforms(img)
		return {'img': img, 'coord': coord}
	
	def to_tf_dataset(self, batch_size=256, shuffle=False):
		"""Convert to TensorFlow Dataset"""
		def generator():
			for idx in range(self.length):
				yield self[idx]
		
		# Transform outputs HWC format (H, W, 3), so signature should match
		# After batching it becomes (B, H, W, 3)
		output_signature = {
			'img': tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
			'coord': tf.TensorSpec(shape=(2,), dtype=tf.int32)
		}
		
		dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
		if shuffle:
			dataset = dataset.shuffle(buffer_size=min(1000, self.length))
		dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
		return dataset

class Dataset_All_Bags:

	def __init__(self, csv_path):
		self.df = pd.read_csv(csv_path)
	
	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		return self.df['slide_id'][idx]




