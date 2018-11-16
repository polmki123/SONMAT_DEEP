from PIL import Image, ImageDraw, ImageFont, ImageFilter
import os

korean_label = []
Font_dir = '../Font_maker/label/'

def makeImage(font_name):
	global Font_dir, korean_label
	
	im = Image.open(font_name)
	img = im.convert('L')
	img.save(os.path.join(font_name))

def main():
	global Font_dir, korean_label

	list_files = os.listdir( Font_dir )
	for i in list_files:
		makeImage(i)

if __name__=='__main__':
	main()

