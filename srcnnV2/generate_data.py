"""Generate training/validation/test data from moon surface image.  This version
is adapted from prepare.py to put the data sets into an HDF5 file.

Image from https://astrogeology.usgs.gov/search/map/Moon/Clementine/UVVIS/Lunar_Clementine_UVVIS_750nm_Global_Mosaic_118m_v2
"""


import argparse
import platform
import datetime
import os

import numpy as np

import h5py
import PIL.Image

# example invocation:
# python generate_data.py ../Lunar_Clementine_UVVIS_750nm_Global_Mosaic_1.2km_v2.1.png 1000 500 500 -u 4 -ip ../images/ -p ../

if __name__ == '__main__':
	# Parse arguments.
	parser = argparse.ArgumentParser(description='Generate training/test/validation images for MoonSRCNN')
	parser.add_argument('file', type=str, help='Mosaic to generate samples from')
	parser.add_argument('num_training_samples', type=int, help='Number of training samples to generate')
	parser.add_argument('num_testing_samples', type=int, help='Number of testing samples to generate')
	parser.add_argument('num_validation_samples', type=int, help='Number of validation samples to generate')
	parser.add_argument('-u', dest='upscale', type=int, default=4, help='Super resolution upscale factor')
	parser.add_argument('-s', dest='sample_size', type=int, default=256, help='Rectangular pixel sample size')
	parser.add_argument('-t', dest='top_cut_fraction', type=float, default=0.2, help='Fraction of the top (NH) of the image to ignore')
	parser.add_argument('-b', dest='bottom_cut_fraction', type=int, default=0.8, help='Fraction of the bottom (SH) of the image to ignore')
	parser.add_argument('-r', dest='seed', type=int, default=18732467, help='Random seed')
	parser.add_argument('-ip', dest='png_output_path', type=str, default=None, help='Path to output PNG versions of the images.')
	parser.add_argument('-p', dest='output_path', type=str, default='./', help='Output path')
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

		# Open the HDF file for writing, setup two datasets (one for the LR
		# images and one for the HR images), and also add some meta data
		# to the dataset so we can go back at a later date and see what
		# was used to generate the data.
		h5_file = h5py.File(os.path.join(args.output_path, key+'.hdf'), 'w')
		h5_lr_data = h5_file.create_dataset('lr', (datasets[key],args.sample_size,args.sample_size),
									chunks=(1,args.sample_size,args.sample_size),
									dtype=np.float32)
		h5_hr_data = h5_file.create_dataset('hr', (datasets[key],args.sample_size,args.sample_size),
									chunks=(1,args.sample_size,args.sample_size),
									dtype=np.float32)
		h5_file.attrs['map'] = args.file
		h5_file.attrs['date'] = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
		h5_file.attrs['platform'] = platform.platform()
		h5_file.attrs['processor'] = platform.processor()
		h5_file.attrs['python_version'] = platform.python_version()
		h5_file.attrs['num_samples'] = datasets[key]
		h5_file.attrs['seed'] = args.seed
		h5_file.attrs['upscale'] = args.upscale
		h5_file.attrs['sample_size'] = args.sample_size
		h5_file.attrs['top_cut_fraction'] = args.top_cut_fraction
		h5_file.attrs['bottom_cut_fraction'] = args.bottom_cut_fraction

		# If the user asked us to also generate a whole set of PNG files
		# then make sure all the directories exist, if they don't, then create
		# them.
		if args.png_output_path is not None:
			if not os.path.isdir(args.png_output_path):
				os.mkdir(args.png_output_path)

			img_path = os.path.join(args.png_output_path,key)
			if not os.path.isdir(img_path):
				os.mkdir(img_path)
		else:
			img_path = None

		# Here we actually generate all the images.
		for i in range(datasets[key]):
			print('Generating {} image {:5d}/{:5d}'.format(key, i+1, datasets[key]))

			# Generate random numbers for start x and y pixel numbers.
			ix = np.random.randint(w - args.sample_size)
			iy = np.random.randint(hprime - args.sample_size) + h0

			# Cut out the high resolution sample (img_hr), generate a low
			# resolution version that has been bicubic upsampled to the original
			# sample size.
			img_hr = img.crop([ix,iy,ix+args.sample_size,iy+args.sample_size])
			img_lr = img_hr.resize((int(args.sample_size/args.upscale),
									int(args.sample_size/args.upscale)),
									resample=PIL.Image.BICUBIC)
			img_lr = img_lr.resize((args.sample_size,args.sample_size),
									resample=PIL.Image.BICUBIC)

			# Write the images to the HDF file.
			h5_lr_data[i,:,:] = np.array(img_lr)[:,:,0].astype(np.float32)
			h5_hr_data[i,:,:] = np.array(img_hr)[:,:,0].astype(np.float32)

			# Save PNGs if we have been asked to do so.
			if args.png_output_path is not None:
				with open(os.path.join(img_path,'{:04d}_hr.png'.format(i)),'wb') as fh:
					img_hr.save(fh)

				with open(os.path.join(img_path,'{:04d}_lr.png'.format(i)),'wb') as fh:
					img_lr.save(fh)

		# Close the HDF file we've just created.
		h5_file.close()
