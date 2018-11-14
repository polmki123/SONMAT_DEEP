from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps
import cv2
from random import *
import os

korean_label = []
Font_dir = '../Font_maker/basic_Font/'

def makeImage(font_name):
	blur_random = randint(3,5)

	img = cv2.imread(Font_dir + font_name)
	im = cv2.blur(img, (blur_random,blur_random))

	im = im.convert('L')
	im.save(os.path.join('./', font_name + '.jpg'))

def main():
	global Font_dir, korean_label
	list_files = os.listdir(Font_dir)
	for i in list_files:
		makeImage(i)

if __name__ == '__main__':
	main()
