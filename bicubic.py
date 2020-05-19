import os
import cv2
import numpy as np
import PIL.Image

hr_dir = '/home/alice/alice/masters_project/MoonSRCNN-master/dataset2/validation/'
bic_dir = '/home/alice/alice/masters_project/MoonSRCNN-master/dataset2/val_bic/'
do_dir = '/home/alice/alice/masters_project/MoonSRCNN-master/dataset2/val_down/'

files = os.listdir(hr_dir)
upscale = 4

for f in files:
	img = PIL.Image.open(hr_dir+f)
	w,h = img.size
	input = img.resize((int(w/upscale), int(h/upscale)), 
						resample=PIL.Image.BICUBIC)
	input.save(do_dir+f)
	input = input.resize((w, h), resample=PIL.Image.BICUBIC)

	input.save(bic_dir+f.replace('.png', '_lr.png'))


        
    


