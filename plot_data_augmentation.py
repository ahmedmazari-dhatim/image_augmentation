from PIL import Image
import glob
image_list = []
for filename in glob.glob('/home/ahmed/Downloads/test/try/*.png'): #assuming gif
    im=Image.open(filename)
    image_list.append(im)
plt.show(im)
