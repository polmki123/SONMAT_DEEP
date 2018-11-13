from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps
import os
import cv2

korean_label = []
img_dir = '../Font_maker/basic_Font/'

def makeImage(img_name):
    img = cv2.imread(img_name)
    blur = cv2.blur(img,(3,3))
    cv2.imshow('blur',blur)
    cv2.waitKey(0)
def main():
	global Font_dir, korean_label
	list_files = os.listdir(Font_dir)
	for i in list_files:
		makeImage(img_dir + i)

if __name__ == '__main__':
	main()

