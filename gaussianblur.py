from __future__ import print_function, division
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug import parameters as iap
#from skimage import
import numpy as np
from scipy import ndimage, misc
from skimage import data
import matplotlib.pyplot as plt
from matplotlib import gridspec
import six
import six.moves as sm
import cv2
from imgaug.parameters import StochasticParameter, Deterministic
from scipy.ndimage import rotate
import os
import sys
import glob


st = lambda aug: iaa.Sometimes(1.0, aug)





def gaussianblur(input):
    image12 = cv2.imread(input)
    #label = input.rsplit('/', 1)[1].rsplit('.', 1)[0]
    h, w = image12.shape[0:2]
    seq12 = st(iaa.GaussianBlur(sigma=1.75))
    #y = seq12.name.rsplit('Unnamed')[1]
    images_aug12=seq12.draw_grid(image12,cols=1,rows=1)
    input=input.rsplit('.')[0]
    misc.imsave('/home/ahmed/Pictures/cogedis/augmented_cogedis/' + str(input) + '_gaussianblur.png', images_aug12)
    print("super")



def main():
    path='/home/ahmed/Pictures/cogedis/cogedis_words_3/'
    os.chdir(path)
    images_name = glob.glob("*.png")
    print("as")
    i=0
    for img in images_name:
        i +=1
        print("ok")
        print(i)
        gaussianblur(img)
        print("sa marche")

if __name__ == "__main__":
    main()