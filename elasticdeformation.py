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
import glob as glob
import sys

st = lambda aug: iaa.Sometimes(1.0, aug)



def elasticdeformation(input):

    image18 = cv2.imread(input)
    #label = input.rsplit('/', 1)[1].rsplit('.', 1)[0]
    h, w = image18.shape[0:2]
    seq18 = st(iaa.ElasticTransformation(alpha=0.75, sigma=0.2))
    #y = seq18.name.rsplit('Unnamed')[1]
    images_aug18=seq18.draw_grid(image18,cols=1,rows=1)
    #images_aug18=seq18.augment_image(image18)
    input = input.rsplit('.')[0]
    misc.imsave('/home/ahmed/Pictures/cogedis/augmented_cogedis/' + str(input) + '_elasticdeformation.png', images_aug18)
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
        elasticdeformation(img)
        print("sa marche")

if __name__ == "__main__":
    main()