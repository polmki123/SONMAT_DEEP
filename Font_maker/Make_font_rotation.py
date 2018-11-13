from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps
from random import *
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
	font_size = 36
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

	im = Image.new("RGB", (width, height), font_color)

	#다
	txt = Image.new("RGB", (70, 70), font_color)
	draw = ImageDraw.Draw(txt)

	font_random_rotation = uniform(0.0, 19.0)

	unicode_font = ImageFont.truetype(os.path.join(Font_dir, font_name), font_size)
	draw.text((10, 0), unicode_text, font=unicode_font, fill=back_ground_color)
	txt = txt.rotate(font_random_rotation)
	im.paste(txt)
	font_total_size += font_size

	#람
	txt = Image.new("RGB", (70, 70), font_color)
	draw = ImageDraw.Draw(txt)

	font_random_rotation = uniform(0.0, 19.0)

	unicode_font = ImageFont.truetype(os.path.join(Font_dir, font_name), font_size)
	draw.text((3, 0), unicode_text2, font=unicode_font, fill=back_ground_color)
	txt = txt.rotate(font_random_rotation)
	im.paste(txt, (font_total_size,0))
	font_total_size += font_size

	#쥐
	txt = Image.new("RGB", (70, 70), font_color)
	draw = ImageDraw.Draw(txt)

	font_random_rotation = uniform(0.0, 19.0)

	unicode_font = ImageFont.truetype(os.path.join(Font_dir, font_name), font_size)
	draw.text((3, 0), unicode_text3, font=unicode_font, fill=back_ground_color)
	txt = txt.rotate(font_random_rotation)
	im.paste(txt, (font_total_size,0))
	font_total_size += font_size

	#space
	space_tatal_size += space_size

	#헌
	txt = Image.new("RGB", (70, 70), font_color)
	draw = ImageDraw.Draw(txt)

	font_random_rotation = uniform(0.0, 19.0)

	unicode_font = ImageFont.truetype(os.path.join(Font_dir, font_name), font_size)
	draw.text((3, 0), unicode_text4, font=unicode_font, fill=back_ground_color)
	txt = txt.rotate(font_random_rotation)
	im.paste(txt, (font_total_size + space_tatal_size,0))
	font_total_size += font_size

	#space
	space_tatal_size += space_size

	#쳇
	txt = Image.new("RGB", (70, 70), font_color)
	draw = ImageDraw.Draw(txt)

	font_random_rotation = uniform(0.0, 19.0)

	unicode_font = ImageFont.truetype(os.path.join(Font_dir, font_name), font_size)
	draw.text((3, 0), unicode_text5, font=unicode_font, fill=back_ground_color)
	txt = txt.rotate(font_random_rotation)
	im.paste(txt, (font_total_size + space_tatal_size, 0))
	font_total_size += font_size

	#바
	txt = Image.new("RGB", (70, 70), font_color)
	draw = ImageDraw.Draw(txt)

	font_random_rotation = uniform(0.0, 19.0)

	unicode_font = ImageFont.truetype(os.path.join(Font_dir, font_name), font_size)
	draw.text((3, 0), unicode_text6, font=unicode_font, fill=back_ground_color)
	txt = txt.rotate(font_random_rotation)
	im.paste(txt, (font_total_size + space_tatal_size, 0))
	font_total_size += font_size
	
	#퀴
	txt = Image.new("RGB", (70, 70), font_color)
	draw = ImageDraw.Draw(txt)

	font_random_rotation = uniform(0.0, 19.0)

	unicode_font = ImageFont.truetype(os.path.join(Font_dir, font_name), font_size)
	draw.text((3 , 0), unicode_text7, font=unicode_font, fill=back_ground_color)
	txt = txt.rotate(font_random_rotation)
	im.paste(txt, (font_total_size + space_tatal_size, 0))
	font_total_size += font_size
	
	#에
	txt = Image.new("RGB", (70, 70), font_color)
	draw = ImageDraw.Draw(txt)

	font_random_rotation = uniform(0.0, 19.0)

	unicode_font = ImageFont.truetype(os.path.join(Font_dir, font_name), font_size)
	draw.text((3 , 0), unicode_text8, font=unicode_font, fill=back_ground_color)
	txt = txt.rotate(font_random_rotation)
	im.paste(txt, (font_total_size + space_tatal_size, 0))
	font_total_size += font_size

	#space
	space_tatal_size += space_size

	#타
	txt = Image.new("RGB", (70, 70), font_color)
	draw = ImageDraw.Draw(txt)

	font_random_rotation = uniform(0.0, 19.0)

	unicode_font = ImageFont.truetype(os.path.join(Font_dir, font_name), font_size)
	draw.text((3 , 0), unicode_text9, font=unicode_font, fill=back_ground_color)
	txt = txt.rotate(font_random_rotation)
	im.paste(txt, (font_total_size + space_tatal_size, 0))
	font_total_size += font_size

	#고
	txt = Image.new("RGB", (70, 70), font_color)
	draw = ImageDraw.Draw(txt)

	font_random_rotation = uniform(0.0, 19.0)

	unicode_font = ImageFont.truetype(os.path.join(Font_dir, font_name), font_size)
	draw.text((3, 0), unicode_text10, font=unicode_font, fill=back_ground_color)
	txt = txt.rotate(font_random_rotation)
	im.paste(txt, (font_total_size + space_tatal_size, 0))
	font_total_size += font_size

	#파
	txt = Image.new("RGB", (70, 70), font_color)
	draw = ImageDraw.Draw(txt)

	font_random_rotation = uniform(0.0, 19.0)

	unicode_font = ImageFont.truetype(os.path.join(Font_dir, font_name), font_size)
	draw.text((3, 0), unicode_text11, font=unicode_font, fill=back_ground_color)
	txt = txt.rotate(font_random_rotation)
	im.paste(txt, (font_total_size + space_tatal_size, 0))
	font_total_size += font_size

	im = ImageOps.invert(im)
	im.save(os.path.join('./', font_name + '.jpg'))

def main():
	global Font_dir, korean_label

	list_files = os.listdir(Font_dir)
	for i in list_files:
		makeImage(i)


if __name__ == '__main__':
	main()

