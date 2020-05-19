"""SRCNN network for moons; heavy influences/borrowing code from https://github.com/basher666/pytorch_srcnn
"""

import argparse
import glob
import os

import torch

import model
import data

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

def progressbar(prefix, x, y, postfix, symbol='■', length=32):
	a = int(length*x/y)
	b = length-a
	print('\r{} [{}{}] {}'.format(prefix,symbol*a,' '*b,postfix), end='\r')

input_folder = '/home/alice/alice/masters_project/MoonSRCNN-master/dataset2/validation'
output_folder = '/home/alice/alice/masters_project/MoonSRCNN-master/dataset2/val_sr'

if __name__ == '__main__':
	# Parse arguments.
	parser = argparse.ArgumentParser(description='MoonSRCNN Validation')
	# parser.add_argument('file', type=str, default = '/home/alice/alice/masters_project/MoonSRCNN-master/dataset2/testing', help='File or wildcard pattern to run the model on')
	parser.add_argument('-u', dest='upscale', type=int, default=4, help='Super resolution upscale factor; default=4')
	parser.add_argument('-m', dest='model', type=str, default='model.pth', help='Model to load')
	args = parser.parse_args()

	model = torch.load(args.model).to(device)

	files = os.listdir(input_folder) 

	# files = glob.glob(args.file)
	i = 0
	for i in range(len(files)):
		f = input_folder+'/'+files[i]

		lr = data.load_input(f, args.upscale).to(device)
		sr = model(torch.reshape(lr, (1,1,lr.shape[1],lr.shape[2])))
		img = data.tensor_to_img(sr.cpu()).convert('RGBA')
		root,ext = os.path.splitext(f)
		with open(output_folder+'/'+'sr_'+files[i], 'wb') as fh:
			img.save(fh)
		progressbar('Processing ', i+1, len(files), '')
