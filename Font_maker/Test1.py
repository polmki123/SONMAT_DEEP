from PIL import Image
import numpy as np
import os 

a = Image.open('1.ttf.jpg')
b = np.array(a)
print(b.shape)