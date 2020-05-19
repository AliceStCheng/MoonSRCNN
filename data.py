"""Utility classes to handle data for MoonSRCNN"""

import glob
import numpy as np

import torch.utils.data
import torch

import PIL.Image
import PIL.ImageFilter

# Heavily influenced by https://github.com/basher666/pytorch_srcnn/blob/master/srcnn_data_utils.py
class DatasetFromFolder(torch.utils.data.Dataset):
	def __init__(self, image_dir, wildcard='*.png', upscale=4):
		super(DatasetFromFolder, self).__init__()
		self.image_filenames = glob.glob(image_dir+'/'+wildcard)
		self._upscale = upscale

	def __getitem__(self, index):
		input = PIL.Image.open(self.image_filenames[index]).convert('L')
		target = input.copy()

		# Input image is bicubic downsampled and then re-upsampled to simulate
		# an input LR image.
		w,h = input.size
		input = input.resize((int(w/self._upscale), int(h/self._upscale)),
							resample=PIL.Image.BICUBIC)
		input = input.resize((w, h), resample=PIL.Image.BICUBIC)
		input = np.asarray(input, dtype=np.float32)/255.0
		input = np.reshape(input, (1, input.shape[0], input.shape[1]))

		target = np.asarray(target, dtype=np.float32)/255.0
		target = np.reshape(target, (1, target.shape[0], target.shape[1]))

		return input, target

	def __len__(self):
		return len(self.image_filenames)

def load_input(filename, upscale):
	img = PIL.Image.open(filename).convert('L')
	w,h = img.size
	input = img.resize((int(w/upscale), int(h/upscale)),
						resample=PIL.Image.BICUBIC)
	input = input.resize((w, h), resample=PIL.Image.BICUBIC)
	input = np.asarray(input, dtype=np.float32)/255.0
	input = np.reshape(input, (1, input.shape[0], input.shape[1]))

	return torch.tensor(input)

def load_target(filename):
	target = PIL.Image.open(filename).convert('L')
	target = np.asarray(target, dtype=np.float32)/255.0
	target = np.reshape(target, (1, 1, target.shape[0], target.shape[1]))
	return torch.tensor(target)

def tensor_to_img(t):
	tmp = np.squeeze(t*255.0)
	tmp[tmp<0]==0
	tmp[tmp>255]==255
	img = PIL.Image.fromarray(np.uint8(tmp.detach().numpy()))
	return img
