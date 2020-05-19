# code from: https://towardsdatascience.com/histograms-in-image-processing-with-skimage-python-be5938962935

# For greyscale - not tested yet.

from skimage import io
import matplotlib.pyplot as plt

import numpy as np

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


# For the original image
image_hr = io.imread('/home/alice/alice/masters_project/MoonSRCNN-master/dataset2/validation/0006_hr.png')
image_sr = io.imread('/home/alice/alice/masters_project/MoonSRCNN-master/dataset2/validation/0006_hr_sr.png')
image_hr = rgb2gray(image_hr)/255 * 100
image_sr = rgb2gray(image_sr)/255 * 100


plt.title('Albedo HR: {:.1f}%    Abedo SR: {:.1f}%'.format(np.mean(image_hr), np.mean(image_sr)))
plt.hist(image_hr.ravel(), bins = 256, color = 'blue', alpha = 0.5)
plt.hist(image_sr.ravel(), bins = 256, color = 'orange', alpha = 0.5)
plt.xlabel('Intensity value')
plt.ylabel('Counts')
plt.tight_layout()
plt.savefig('../outputs/histo_monochrome.png')
# plt.show()

# Zoomed in version of above
plt.title('Albedo HR: {:.1f}%    Abedo SR: {:.1f}%'.format(np.mean(image_hr), np.mean(image_sr)))
plt.hist(image_hr.ravel(), bins = 256, color = 'blue', alpha = 0.5)
plt.hist(image_sr.ravel(), bins = 256, color = 'orange', alpha = 0.5)
plt.xlim([0,200])
plt.ylim([0,15000])
plt.xlabel('Intensity value')
plt.ylabel('Counts')
plt.tight_layout()
plt.savefig('../outputs/histo_mono_100_bins.png')
# plt.show()


# for a 3 channel RGB image
# image = io.imread('/home/alice/alice/masters_project/MoonSRCNN-master/dataset2/validation/0006_hr.png')
# _ = plt.hist(image.ravel(), bins = 256, color = 'orange', )
# _ = plt.hist(image[:, :, 0].ravel(), bins = 256, color = 'red', alpha = 0.5)
# _ = plt.hist(image[:, :, 1].ravel(), bins = 256, color = 'Green', alpha = 0.5)
# _ = plt.hist(image[:, :, 2].ravel(), bins = 256, color = 'Blue', alpha = 0.5)
# _ = plt.xlabel('Intensity Value')
# _ = plt.ylabel('Count')
# _ = plt.legend(['Total', 'Red_Channel', 'Green_Channel', 'Blue_Channel'])
# plt.savefig('../outputs/histo_rgb.png')
# # plt.show()