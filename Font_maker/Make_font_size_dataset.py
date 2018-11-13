from PIL import Image, ImageDraw, ImageFont, ImageFilter
from random import*
import os
# import cv2

korean_label = []
Font_dir = '../Font_maker/Font/'

def makeImage(font_name):
	global Font_dir, korean_label
	if not os.path.exists(font_name):
		os.makedirs(font_name)

	# configuration
	width = 512
	height = 64
	back_ground_color = (255, 255, 255)
	font_color = (0, 0, 0)
	font_total_size = 10
	space_size = 10
	space_tatal_size = 0

	unicode_text = u"다"
	unicode_text2 = u"람"
	unicode_text3 = u"쥐"
	unicode_space = u" "
	unicode_text4 = u"헌"
	unicode_space2 = u" "
	unicode_text5 = u"쳇"
	unicode_text6 = u"바"
	unicode_text7 = u"퀴"
	unicode_text8 = u"에"
	unicode_space3 = u" "
	unicode_text9 = u"타"
	unicode_text10 = u"고"
	unicode_text11 = u"파"


	im = Image.new("RGB", (width, height), back_ground_color)
	draw = ImageDraw.Draw(im)

	#다
	font_random_size = randint(24, 48)
	unicode_font = ImageFont.truetype(os.path.join(Font_dir, font_name), font_random_size)
	draw.text((10, 0), unicode_text, font=unicode_font, fill=font_color)
	font_total_size += font_random_size

	#람
	font_random_size = randint(24, 48)
	unicode_font = ImageFont.truetype(os.path.join(Font_dir, font_name), font_random_size)
	draw.text((font_total_size, 0), unicode_text2, font=unicode_font, fill=font_color)
	font_total_size += font_random_size

	#쥐
	font_random_size = randint(24, 48)
	unicode_font = ImageFont.truetype(os.path.join(Font_dir, font_name), font_random_size)
	draw.text((font_total_size, 0), unicode_text3, font=unicode_font, fill=font_color)
	font_total_size += font_random_size

	#space
	space_tatal_size += space_size

	#헌
	font_random_size = randint(24, 48)
	unicode_font = ImageFont.truetype(os.path.join(Font_dir, font_name), font_random_size)
	draw.text((font_total_size + space_tatal_size, 0), unicode_text4, font=unicode_font, fill=font_color)
	font_total_size += font_random_size

	#space
	space_tatal_size += space_size

	#쳇
	font_random_size = randint(24, 48)
	unicode_font = ImageFont.truetype(os.path.join(Font_dir, font_name), font_random_size)
	draw.text((font_total_size + space_tatal_size, 0), unicode_text5, font=unicode_font, fill=font_color)
	font_total_size += font_random_size

	#바
	font_random_size = randint(24, 48)
	unicode_font = ImageFont.truetype(os.path.join(Font_dir, font_name), font_random_size)
	draw.text((font_total_size + space_tatal_size, 0), unicode_text6, font=unicode_font, fill=font_color)
	font_total_size += font_random_size

	#퀴
	font_random_size = randint(24, 48)
	unicode_font = ImageFont.truetype(os.path.join(Font_dir, font_name), font_random_size)
	draw.text((font_total_size + space_tatal_size, 0), unicode_text7, font=unicode_font, fill=font_color)
	font_total_size += font_random_size

	#에
	font_random_size = randint(24, 48)
	unicode_font = ImageFont.truetype(os.path.join(Font_dir, font_name), font_random_size)
	draw.text((font_total_size + space_tatal_size, 0), unicode_text8, font=unicode_font, fill=font_color)
	font_total_size += font_random_size

	#space
	space_tatal_size += space_size

	#타
	font_random_size = randint(24, 48)
	unicode_font = ImageFont.truetype(os.path.join(Font_dir, font_name), font_random_size)
	draw.text((font_total_size + space_tatal_size, 0), unicode_text9, font=unicode_font, fill=font_color)
	font_total_size += font_random_size

	#고
	font_random_size = randint(24, 48)
	unicode_font = ImageFont.truetype(os.path.join(Font_dir, font_name), font_random_size)
	draw.text((font_total_size + space_tatal_size, 0), unicode_text10, font=unicode_font, fill=font_color)
	font_total_size += font_random_size

	#파
	font_random_size = randint(24, 48)
	unicode_font = ImageFont.truetype(os.path.join(Font_dir, font_name), font_random_size)
	draw.text((font_total_size + space_tatal_size, 0), unicode_text11, font=unicode_font, fill=font_color)
	font_total_size += font_random_size

	im.save(os.path.join('./', font_name + '.jpg'))

def main():
	global Font_dir, korean_label

	list_files = os.listdir(Font_dir)
	for i in list_files:
		makeImage(i)

if __name__ == '__main__':
	main()

