import cv2
import numpy as np
import sys
import PIL.Image

img = PIL.Image.open("/home/alice/alice/masters_project/MoonSRCNN-master/dataset2/validation/0006_hr.png").convert('L')

img_bic_down = img.copy()

w,h = img_bic_down.size
img_bic_down.show() # show original image

img_bic_down = img_bic_down.resize((int(w/4), int(h/4)),
					resample=PIL.Image.BICUBIC)
# img_bic_down = img_bic_down.resize((w, h), resample=PIL.Image.BICUBIC)
img_bic_down.show()

# resize back up using biciubic for comparison

# img_bic_up = img_bic_down.resize((int(w*4), int(h*4)),resample=PIL.Image.BICUBIC)
img_bic_up = img_bic_down.resize((w, h), resample=PIL.Image.BICUBIC)


# img_bic_down.show()
img_bic_up.show()
img_bic_down = img_bic_down.save("bicubic_downsample.png")
img_bic_up = img_bic_up.save("bicubic_reupsample.png")

# work out why down and up sample look the same
