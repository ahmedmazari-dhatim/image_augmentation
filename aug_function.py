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
import glob
import os
import sys


st = lambda aug: iaa.Sometimes(0.3, aug)

class BinomialRows(StochasticParameter):
    def __init__(self, p):
        super(BinomialRows, self).__init__()

        if isinstance(p, StochasticParameter):
            self.p = p
        elif ia.is_single_number(p):
            assert 0 <= p <= 1.0, "Expected probability p to be in range [0.0, 1.0], got %s." % (p,)
            self.p = Deterministic(float(p))
        else:
            raise Exception("Expected StochasticParameter or float/int value, got %s." % (type(p),))

    def _draw_samples(self, size, random_state):
        p = self.p.draw_sample(random_state=random_state)
        assert 0 <= p <= 1.0, "Expected probability p to be in range [0.0, 1.0], got %s." % (p,)
        h, w, c = size
        drops = random_state.binomial(1, p, (h, 1, c))
        drops_rows = np.tile(drops, (1, w, 1))
        return drops_rows

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        if isinstance(self.p, float):
            return "BinomialRows(%.4f)" % (self.p,)
        else:
            return "BinomialRows(%s)" % (self.p,)

class BinomialColumns(StochasticParameter):
    def __init__(self, p):
        super(BinomialColumns, self).__init__()

        if isinstance(p, StochasticParameter):
            self.p = p
        elif ia.is_single_number(p):
            assert 0 <= p <= 1.0, "Expected probability p to be in range [0.0, 1.0], got %s." % (p,)
            self.p = Deterministic(float(p))
        else:
            raise Exception("Expected StochasticParameter or float/int value, got %s." % (type(p),))

    def _draw_samples(self, size, random_state):
        p = self.p.draw_sample(random_state=random_state)
        assert 0 <= p <= 1.0, "Expected probability p to be in range [0.0, 1.0], got %s." % (p,)
        h, w, c = size
        drops = random_state.binomial(1, p, (1, w, c))
        drops_columns = np.tile(drops, (h, 1, 1))
        return drops_columns

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        if isinstance(self.p, float):
            return "BinomialColumns(%.4f)" % (self.p,)
        else:
            return "BinomialColumns(%s)" % (self.p,)




def draw_horizontal_line(input):

    image1=cv2.imread(input)
    #label=input.rsplit('/',1)[1].rsplit('.',1)[0]
    h,w=image1.shape[0:2]
    horizontal_line=cv2.line(image1,(0,int(h/2)),(w,int(h/2)),(0,0,0),2)
    #cv2.imshow("image",horizontal_line)
    #cv2.waitKey(0)
    input = input.rsplit('.')[0]
    cv2.imwrite('/home/ahmed/Pictures/cogedis/24072017/all_augmented/'+str(input)+'_horizontalline.png',horizontal_line)
    cv2.destroyAllWindows()

    #print("super")

def draw_diagonal_line(input):

    image2=cv2.imread(input)
    #label=input.rsplit('/',1)[1].rsplit('.',1)[0]
    h,w=image2.shape[0:2]
    diagonal_line= cv2.line(image2,(0,0),(w,h),(0,0,0),2)
    input = input.rsplit('.')[0]
    cv2.imwrite('/home/ahmed/Pictures/cogedis/24072017/all_augmented/'+str(input)+'_diagonalline.png', diagonal_line)
    cv2.destroyAllWindows()

    #print("super")

def draw_diagonal_inverse_line(input):

    image3=cv2.imread(input)
    #label=input.rsplit('/',1)[1].rsplit('.',1)[0]
    h,w=image3.shape[0:2]
    diagonal_inverse_line= cv2.line(image3,(0,h),(w,0),(0,0,0),2)
    input = input.rsplit('.')[0]
    cv2.imwrite('/home/ahmed/Pictures/cogedis/24072017/all_augmented/'+str(input)+'_diagonalinverseline.png', diagonal_inverse_line)
    cv2.destroyAllWindows()

    #print("super")

def draw_vertical_left_line(input):

    image4=cv2.imread(input)
    #label=input.rsplit('/',1)[1].rsplit('.',1)[0]
    h,w=image4.shape[0:2]
    vertical_left_line= cv2.line(image4,(0,0),(0,h),(0,0,0),4)
    input = input.rsplit('.')[0]
    cv2.imwrite('/home/ahmed/Pictures/cogedis/24072017/all_augmented/'+str(input)+'_verticalleftline.png', vertical_left_line)
    cv2.destroyAllWindows()

    #print("super")

def draw_vertical_right_line(input):

    image5 = cv2.imread(input)
    #label = input.rsplit('/', 1)[1].rsplit('.', 1)[0]
    h, w = image5.shape[0:2]
    vertical_right_line = cv2.line(image5, (w, 0), (w,h), (0, 0, 0), 4)
    input = input.rsplit('.')[0]
    cv2.imwrite('/home/ahmed/Pictures/cogedis/24072017/all_augmented/'+str(input)+'_verticalrightline.png', vertical_right_line)
    cv2.destroyAllWindows()

    #print("super")

def draw_several_rows_line(input):

    image6 = cv2.imread(input)
    #label = input.rsplit('/', 1)[1].rsplit('.', 1)[0]
    h, w = image6.shape[0:2]
    p_row = 0.2
    aug = iaa.Sequential([
        iaa.Dropout(p=BinomialRows(1 - p_row))])
    new_image=aug.augment_image(image6)
    input = input.rsplit('.')[0]
    misc.imsave('/home/ahmed/Pictures/cogedis/24072017/all_augmented/'+str(input)+'_severalrowsline.png', new_image)

    #print("super")

def draw_several_cols_line(input):
    image7 = cv2.imread(input)
    #label = input.rsplit('/', 1)[1].rsplit('.', 1)[0]
    h, w = image7.shape[0:2]
    p_column = 0.2
    aug7 = iaa.Sequential([ iaa.Dropout(p=BinomialColumns(1 - p_column))])
    new_image7=aug7.augment_image(image7)
    input = input.rsplit('.')[0]
    misc.imsave('/home/ahmed/Pictures/cogedis/24072017/all_augmented/'+str(input)+'_severalcolsline.png', new_image7)
    #print("super")
def draw_several_cols_rows_line(input):
    image8 = cv2.imread(input)
    #label = input.rsplit('/', 1)[1].rsplit('.', 1)[0]
    h, w = image8.shape[0:2]
    p_row = 0.2
    p_column = 0.2
    aug8 = iaa.Sequential([
        iaa.Dropout(p=BinomialRows(1 - p_row)),
        iaa.Dropout(p=BinomialColumns(1 - p_column)),
    ])
    new_image8=aug8.augment_image(image8)
    input = input.rsplit('.')[0]
    misc.imsave('/home/ahmed/Pictures/cogedis/24072017/all_augmented/'+str(input)+'_severalcolsrowsline.png', new_image8)
    #print("super")



def elasticdeformation(input):

    image18 = cv2.imread(input)
    #label = input.rsplit('/', 1)[1].rsplit('.', 1)[0]
    h, w = image18.shape[0:2]
    seq18 = st(iaa.ElasticTransformation(alpha=0.75, sigma=0.2))
    #y = seq18.name.rsplit('Unnamed')[1]
    images_aug18=seq18.draw_grid(image18,cols=1,rows=1)
    #images_aug18=seq18.augment_image(image18)
    input = input.rsplit('.')[0]
    misc.imsave('/home/ahmed/Pictures/cogedis/24072017/all_augmented/' + str(input) + '_elasticdeformation.png', images_aug18)
    #print("super")



'''
def flipud(input):
    image9 = cv2.imread(input)
    #label = input.rsplit('/', 1)[1].rsplit('.', 1)[0]
    h, w = image9.shape[0:2]
    seq9 = iaa.Flipud(1.0)
    y = seq9.name.rsplit('Unnamed')[1]
    # images_aug = seq.augment_images(image)
    images_aug9 = seq9.draw_grid(image9, cols=1, rows=1)
    input = input.rsplit('.')[0]
    misc.imsave('/home/ahmed/Pictures/cogedis/augmented_cogedis/' + str(input) + '_flipud.png', images_aug9)


    #print("super")
'''
'''
def fliplr(input):
    image10 = cv2.imread(input)
    #label = input.rsplit('/', 1)[1].rsplit('.', 1)[0]
    h, w = image10.shape[0:2]
    seq10 = iaa.Fliplr(1.0)
    y = seq10.name.rsplit('Unnamed')[1]
    #images_aug = seq.augment_images(image)
    images_aug10=seq10.draw_grid(image10,cols=1,rows=1)
    input = input.rsplit('.')[0]
    misc.imsave('/home/ahmed/Pictures/cogedis/augmented_cogedis/' + str(input) + '_fliplr.png', images_aug10)
    #print("super")

'''
def superpixel(input):
    image11 = cv2.imread(input)
    #label = input.rsplit('/', 1)[1].rsplit('.', 1)[0]
    h, w = image11.shape[0:2]
    seq11 = st(iaa.Superpixels(p_replace=0.63))
    y = seq11.name.rsplit('Unnamed')[1]
    images_aug11=seq11.draw_grid(image11,cols=1,rows=1)
    input = input.rsplit('.')[0]
    misc.imsave('/home/ahmed/Pictures/cogedis/24072017/all_augmented/' + str(input) + '_superpixel.png', images_aug11)
    #print("super")

def gaussianblur(input):
    image12 = cv2.imread(input)
    #label = input.rsplit('/', 1)[1].rsplit('.', 1)[0]
    h, w = image12.shape[0:2]
    seq12 = st(iaa.GaussianBlur(sigma=1.75))
    y = seq12.name.rsplit('Unnamed')[1]
    images_aug12=seq12.draw_grid(image12,cols=1,rows=1)
    input = input.rsplit('.')[0]
    misc.imsave('/home/ahmed/Pictures/cogedis/24072017/all_augmented/' + str(input) + '_gaussianblur.png', images_aug12)
    #print("super")



def additivegaussiannoise(input):



    image14 = cv2.imread(input)
    #label = input.rsplit('/', 1)[1].rsplit('.', 1)[0]
    h, w = image14.shape[0:2]
    seq14 = st(iaa.AdditiveGaussianNoise(loc=0, scale=0.40 * 255))
    y = seq14.name.rsplit('Unnamed')[1]
    images_aug14=seq14.draw_grid(image14,cols=1,rows=1)
    input = input.rsplit('.')[0]
    misc.imsave('/home/ahmed/Pictures/cogedis/24072017/all_augmented/' + str(input) + '_additivegaussiannoise.png', images_aug14)
    #print("super")



def dropout(input):

    image15 = cv2.imread(input)
    #label = input.rsplit('/', 1)[1].rsplit('.', 1)[0]
    h, w = image15.shape[0:2]
    seq15 = st(iaa.Dropout(0.6))
    y = seq15.name.rsplit('Unnamed')[1]
    images_aug15=seq15.draw_grid(image15,cols=1,rows=1)
    input = input.rsplit('.')[0]
    misc.imsave('/home/ahmed/Pictures/cogedis/24072017/all_augmented/' + str(input) + '_dropout.png', images_aug15)
    #print("super")

def translation(input):

    image16 = cv2.imread(input)
    #label = input.rsplit('/', 1)[1].rsplit('.', 1)[0]
    h, w = image16.shape[0:2]
    seq16 = st(iaa.Affine(translate_px={"x": (6, 6), "y": (6, 6)}))
    y = seq16.name.rsplit('Unnamed')[1]
    images_aug16=seq16.draw_grid(image16,cols=1,rows=1)
    input = input.rsplit('.')[0]
    misc.imsave('/home/ahmed/Pictures/cogedis/24072017/all_augmented/' + str(input) + '_translation.png', images_aug16)
    #print("super")

def rotation(input,r):

    image17 = cv2.imread(input)
    #label = input.rsplit('/', 1)[1].rsplit('.', 1)[0]
    h, w = image17.shape[0:2]
    seq17 = rotate(image17, r, reshape=True)
    input = input.rsplit('.')[0]
    misc.imsave('/home/ahmed/Pictures/cogedis/24072017/all_augmented/'+ str(input) + '_rotation'+str(r)+'.png', seq17)
    #print("super")




def sharpen(input):
    image13 = cv2.imread(input)
    #label = input.rsplit('/', 1)[1].rsplit('.', 1)[0]
    h, w = image13.shape[0:2]
    seq13 = st(iaa.Sharpen(alpha=1.0, lightness=0.15))
    #y = seq13.name.rsplit('Unnamed')[1]
    images_aug13=seq13.draw_grid(image13,cols=1,rows=1)
    input = input.rsplit('.')[0]
    misc.imsave('/home/ahmed/Pictures/cogedis/24072017/all_augmented/' + str(input) + '_sharpen.png', images_aug13)
    #
    # print("super")



'''
def rotation(input,r):
    image = cv2.imread(input)
    label = input.rsplit('/', 1)[1].rsplit('.', 1)[0]
    h, w = image.shape[0:2]
    seq = st(iaa.Affine(rotate=int(r)))
    y = seq.name.rsplit('Unnamed')[1]
+    images_aug=seq.draw_grid(image,cols=1,rows=1)
    misc.imsave('/home/ahmed/Downloads/test/try/' + str(label) + '_rotation'+str(r)+'.png', images_aug)
    #print("super")

def rotation90(input,r):
    image = cv2.imread(input)
    label = input.rsplit('/', 1)[1].rsplit('.', 1)[0]
    h, w = image.shape[0:2]
    seq = iaa.Sequential([iaa.Rot90(r)])
    y = seq.name.rsplit('Unnamed')[1]
    images_aug=seq.draw_grid(image,cols=1,rows=1)
    misc.imsave('/home/ahmed/Downloads/test/try/' + str(label) + '_rotation'+str(r)+'.png', images_aug)
    #print("super")
'''




def main_1():

    path='/home/ahmed/Pictures/cogedis/24072017/all/'
    os.chdir(path)
    images_name = glob.glob("*.png")
    #print("as")
    for img in images_name:
        print(img)

        #print("ok")
        #draw_several_cols_line(img)
        #draw_several_rows_line(img)
        draw_horizontal_line(img)
        #draw_diagonal_line(img)
        #draw_diagonal_inverse_line(img)
        draw_vertical_left_line(img)
        draw_vertical_right_line(img)
        #draw_several_cols_rows_line(img)
        translation(img)
        superpixel(img)
        additivegaussiannoise(img)
        elasticdeformation(img)
        gaussianblur(img)
        sharpen(img)
        #flipud(img)
        #fliplr(img)
        #rotation(img, 90)
        #rotation(img, -90)
        rotation(img, 5)
        rotation(img, -5)
        dropout(img)


'''

def main_2():
    path = '/home/ahmed/Downloads/test/brut_image/number.png'
    sharpen(path)

def main_3():
    z = '/home/ahmed/Downloads/test/brut_image/number.png'
    gaussianblur(z)

def main_4():
    input_path = '/home/ahmed/Downloads/test/brut_image/number.png'
    elasticdeformation(input_path)
'''





if __name__ == "__main__":
    #main_5()
    #main_4()
    main_1()
    #main_3()
    #main_2()





