from PIL import Image
import numpy as np
import os 

dir = '../SONMAT_DEEP/Font_maker'

#a = Image.open(os.path.join(dir,'126.ttf.jpg'))
a = Image.open('126.ttf.jpg')
b = np.array(a)
print(b.shape)