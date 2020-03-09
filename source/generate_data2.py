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
#	parser.add_argument('num_validation_samples', type=int, help='Number of validation samples to generate')
	parser.add_argument('-u', dest='upscale', type=int, default=4, help='Super resolution upscale factor')
	parser.add_argument('-s', dest='sample_size', type=int, default=128, help='Rectangular pixel sample size')
	parser.add_argument('-t', dest='top_cut_fraction', type=float, default=0.2, help='Fraction of the top (NH) of the image to ignore')
	parser.add_argument('-b', dest='bottom_cut_fraction', type=int, default=0.8, help='Fraction of the bottom (SH) of the image to ignore')
	parser.add_argument('-r', dest='seed', type=int, default=18732467, help='Random seed')
	parser.add_argument('-f', dest='output_path', type=str, default='../dataset/', help='Path to output files.')
	args = parser.parse_args()

	datasets = {'train':args.num_training_samples,
				'test':args.num_testing_samples}

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
		rmse = np.zeros(datasets[key])
		psnr = np.zeros(datasets[key])
		for i in range(datasets[key]):
			print('Generating {} image {:5d}/{:5d}'.format(key, i+1, datasets[key]))

			# Generate random numbers for start x and y pixel numbers.
			ix = np.random.randint(w - args.sample_size)
			iy = np.random.randint(hprime - args.sample_size) + h0

			# Cut out the high resolution sample and generate a low resolution
			# version for further use.
			img_hr = img.crop([ix,iy,ix+args.sample_size,iy+args.sample_size])
			img_lr = img_hr.resize((int(args.sample_size/args.upscale),
									int(args.sample_size/args.upscale)),
									resample=PIL.Image.BICUBIC)
			img_lrup = img_lr.resize((int(args.sample_size),
									int(args.sample_size)),
									resample=PIL.Image.BICUBIC)
			data_hr = np.asarray(img_hr, dtype=np.float32)[:,:,0]
			data_hr /= 255.0
			data_lrup = np.asarray(img_lrup, dtype=np.float32)[:,:,0]
			data_lrup /= 255.0

			# Save.
			filename = args.output_path+key+'_hr/{:04d}_hr.np'.format(i)
			np.save(filename, data_hr)
			filename = args.output_path+key+'_lr/{:04d}_lr.np'.format(i)
			np.save(filename, data_lrup)

			rmse[i] = np.sqrt(np.mean((data_hr - data_lrup)**2))
			psnr[i] = np.log(1 / rmse[i])/np.log(10)*20

		print('Median loss: {}'.format(np.median(rmse)))
		print('Median PNSR: {}'.format(np.median(psnr)))
