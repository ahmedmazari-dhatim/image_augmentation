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
from scipy.ndimage import rotate


print("noublie de rajouter des lignes horizontales et verticales sur les sequeneces)")
image = ndimage.imread("/home/ahmed/Downloads/test/brut_image/number.png")
print(image.shape)

st = lambda aug: iaa.Sometimes(0.3, aug)

#seq=iaa.Sequential([iaa.Flipud(0.5)])  # correct
#seq=iaa.Sequential([iaa.Fliplr(0.5)])  # correct
# seq=st(iaa.Superpixels(p_replace=0.63)) # correct
#seq=st(iaa.GaussianBlur(sigma=1.75))  # correct
#seq=st(iaa.Sharpen(alpha=1.0, lightness=0.15)) # correct
#seq=st(iaa.AdditiveGaussianNoise(loc=0, scale=0.35*255)) # correct elementwise
#seq=st(iaa.Dropout(0.4)) # correct elementwise
#seq= st(iaa.Affine(translate_px={"x": (5, 5), "y": (5, 5)})) # correct translate affine
#seq=st(iaa.Affine(rotate=5)) #coorect affine
#seq=st(iaa.Affine(rotate=-5)) #coorect affine
#seq=st(iaa.Affine(rotate=-90)) # not working affine
#seq=st(iaa.Affine(rotate=90)) # not working affine
#seq=st(iaa.ElasticTransformation(alpha=0.75, sigma=0.2)) # correct
#seq=iaa.Sequential([iaa.Rot90(0.5)])
seq= iaa.Rot90(0.3)
print(seq.name)


y=seq.name.rsplit('Unnamed')[1]
print(y)

#y_st=seq.then_list[0].name.rsplit('Unnamed')[1]
#y=seq.name.rsplit('Unnamed')[1]



images_aug = seq.augment_images([image])
seq.show_grid(image,cols=1,rows=1)
#grid = seq.draw_grid(image, cols=1, rows=1)
#misc.imsave("/home/ahmed/Downloads/test/number_" + str(y) + ".png", grid)

'''
if y_st=='Superpixels':
    misc.imsave("/home/ahmed/Downloads/test/number_replace"+str(y_st)+".png", grid)
if y_st=='Convolve':
    misc.imsave("/home/ahmed/Downloads/test/number_sharpen" + str(y_st) + ".png", grid)
else:
    misc.imsave("/home/ahmed/Downloads/test/number_" + str(y_st) + ".png", grid)

'''
#misc.imsave("/home/ahmed/Downloads/test/number_"+str(y)+".png", grid)
