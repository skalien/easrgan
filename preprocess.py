import numpy as np, skimage.filters, skimage.color
from scipy import misc
import os, re, multiprocessing, sys, warnings
from tqdm import tqdm
from PIL import Image

# warnings.filterwarnings('error')


def process_images(file):
    
    im = Image.open('path' + file)
    im.thumbnail((im.size[0] // 8, im.size[1] // 8), Image.ANTIALIAS)
    
    shape_full_cut = [im.size[0] - im.size[0]%2, im.size[1] - im.size[1]%2]
    shape_one_cut = [shape_full_cut[0]//2, shape_full_cut[1]//2]
    num_cuts = [shape_full_cut[0]//shape_one_cut[0], shape_full_cut[1]//shape_one_cut[1]]
    imcuts = []
    for i in range(num_cuts[0]):
        for j in range(num_cuts[1]):
            imcuts.append(im.crop((i*shape_one_cut[0],
                                   j*shape_one_cut[1],
                                   (i+1)*shape_one_cut[0],
                                   (j+1) * shape_one_cut[1],
                                   )))

    for i, data in enumerate(imcuts):            
        data.save('path2' + file[:-4] + '_' + str(i) + '.png', 'PNG')

def check_files(file):

    try:
        im = Image.open(file, mode='r')
    except:
        os.system('rm ' + file)
        return
    if len(im.getbands()) != 3:
        im.close()
        os.system('rm ' + file)
        return
    if im.size[0] < 96 or im.size[1] < 96:
        im.close()
        os.system('rm ' + file)
        return

def read_imfile(file):

    try:
        misc.imread(file)
    except:
        os.system('rm ' + file)
        print(file)

def convertToLuma(file):

    im = np.array(misc.imread(src + file), dtype=np.float32)
    im_y = 0.299*im[...,0] + 0.587*im[...,1] + 0.114*im[...,2]
    im_y = np.array(im_y, dtype=np.uint8)

    misc.imsave(dst + file, im_y)

    print(file)


tqdm.monitor_interval = 0

src = '/home/bishshoy/ssd/DF2K/1/'
dst = '/home/bishshoy/ssd/DF2K_LUMA/1/'

files = os.listdir(src)

multiprocessing.Pool(12).map(convertToLuma, files)






















