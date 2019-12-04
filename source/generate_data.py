"""Generate training/validation/test data from moon surface image.

Image from https://astrogeology.usgs.gov/search/map/Moon/Clementine/UVVIS/Lunar_Clementine_UVVIS_750nm_Global_Mosaic_118m_v2
"""

import PIL.Image
import numpy as np
import argparse


if __name__ == '__main__':
	# Parse arguments.
	parser = argparse.ArgumentParser(description='Generate training/test/validation images for MoonSRCNN')
	parser.add_argument('file', type=str, help='Mosaic to generate samples from')
	parser.add_argument('num_training_samples', type=int, help='Number of training samples to generate')
	parser.add_argument('num_testing_samples', type=int, help='Number of testing samples to generate')
	parser.add_argument('num_validation_samples', type=int, help='Number of validation samples to generate')
	parser.add_argument('-u', dest='upscale', type=int, default=4, help='Super resolution upscale factor')
	parser.add_argument('-s', dest='sample_size', type=int, default=128, help='Rectangular pixel sample size')
	parser.add_argument('-t', dest='top_cut_fraction', type=float, default=0.2, help='Fraction of the top (NH) of the image to ignore')
	parser.add_argument('-b', dest='bottom_cut_fraction', type=int, default=0.8, help='Fraction of the bottom (SH) of the image to ignore')
	parser.add_argument('-r', dest='seed', type=int, default=18732467, help='Random seed')
	parser.add_argument('-f', dest='output_path', type=str, default='../dataset/', help='Path to output files.')
	parser.add_argument('-l', dest='gen_lr', type=bool, default=False, help='Generate low resolution (LR) counterpart images')
	args = parser.parse_args()

	datasets = {'training':args.num_training_samples,
				'testing':args.num_testing_samples,
				'validation':args.num_validation_samples}

	# Set the random seed for reproducibility.
	np.random.seed(args.seed)

	# Load global mosaic image
	img = PIL.Image.open(args.file)
	w, h = img.size
	h0 = int(h*args.top_cut_fraction)
	h1 = int(h*args.bottom_cut_fraction)
	hprime = h1-h0

	# Generate the samples.
	for key in datasets.keys():
		for i in range(datasets[key]):
			print('Generating {} image {:5d}/{:5d}'.format(key, i+1, datasets[key]))

			# Generate random numbers for start x and y pixel numbers.
			ix = np.random.randint(w - args.sample_size)
			iy = np.random.randint(hprime - args.sample_size) + h0

			# Cut out the high resolution sample and generate a low resolution
			# version for further use.
			img_hr = img.crop([ix,iy,ix+args.sample_size,iy+args.sample_size])
			if args.gen_lr:
				img_lr = img_hr.resize((int(args.sample_size/args.upscale),
										int(args.sample_size/args.upscale)),
										resample=PIL.Image.BICUBIC)

			# Save.
			filename = args.output_path+key+'/{:04d}_hr.png'.format(i)
			with open(filename,'wb') as fh:
				img_hr.save(fh)
			if args.gen_lr:
				filename = args.output_path+key+'/{:04d}_lr.png'.format(i)
				with open(filename,'wb') as fh:
					img_lr.save(fh)
