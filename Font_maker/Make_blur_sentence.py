from PIL import Image, ImageDraw, ImageFont, ImageFilter
from random import *
import os
import cv2

korean_label = []
Font_dir = '../Font_maker/space_size/'

def makeImage(font_name):
	blur_random = randint(2,4)

	img = cv2.imread(Font_dir+font_name)
	img = cv2.blur(img, (blur_random,blur_random))

	cv2.imwrite(font_name, img)

def main():
	global Font_dir, korean_label

	list_files = os.listdir(Font_dir)
	for i in list_files:
		makeImage(i)

if __name__ == '__main__':
	main()
