import numpy as np
import math
import cv2
from skimage.measure import compare_ssim
import matplotlib.pyplot as plt

original = cv2.imread('../../Docs/doc_images/0007_hr.png')
#print(original)
contrast = cv2.imread('../../Docs/doc_images/0007_hr_srcnn_x4.png',1)
#print(contrast)

def mse(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    return mse

def psnr(img1, img2):

    m = mse(img1, img2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(m))

def ssim(img1, img2):
    s = compare_ssim(img1, img2, multichannel=True)
    return s

m = mse(original, contrast)
print('mse is: ', m)
d = ssim(original, contrast)
print('SSIM is: ', d)
p  = psnr(original,contrast)
print('PSNR is: ', p)

def compare_metrics(img1, img2, title):
    m = mse(original, contrast)
    d = ssim(original, contrast)
    p  = psnr(original,contrast)

    fig = plt.figure(title)
    plt.suptitle('MSE: %.2f, PSNR: %.2f SSIM: %.2f' % (m,p,d))

    #for the first images
    ax = fig.add_subplot(1,2,1)
    plt.imshow(img1, cmap = plt.cm.gray)
    plt.axis("off")

    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(img2, cmap = plt.cm.gray)
    plt.axis("off")

    plt.show()

compare_metrics(original, contrast, 'first comparison')
