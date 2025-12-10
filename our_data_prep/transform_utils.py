import tensorflow as tf
import numpy as np
from PIL import Image

def get_eval_transforms(mean, std, target_img_size = -1):

	mean_tensor = tf.constant(mean, dtype=tf.float32)
	std_tensor = tf.constant(std, dtype=tf.float32)
	
	def transform(img):
		img_array = np.array(img, dtype=np.float32)
		
		if target_img_size > 0:
			img_tensor = tf.convert_to_tensor(img_array)
			img_tensor = tf.image.resize(img_tensor, [target_img_size, target_img_size])
		else:
			img_tensor = tf.convert_to_tensor(img_array)
		
		img_tensor = img_tensor / 255.0
		
		img_tensor = (img_tensor - mean_tensor) / std_tensor
		
		return img_tensor
	
	return transform