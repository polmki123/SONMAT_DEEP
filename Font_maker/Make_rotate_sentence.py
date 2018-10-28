from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps
import os
# import cv2

korean_label = []
Font_dir = '../Font_maker/Font/'

def makeImage(font_name):
	global Font_dir, korean_label
	if not os.path.exists(font_name):
		os.makedirs(font_name)

	# configuration

	width = 900
	height = 70
	back_ground_color = (255, 255, 255)
	space_size = 10
	space_size2 = 15
	space_size3 = 25
	font_size = 36
	font_size2 = 45
	font_size3 = 55
	font_size4 = 60
	font_size5 = 60
	font_color = (0, 0, 0)


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
	unicode_text9 = u"타"
	unicode_text10 = u"고"
	unicode_text11 = u"파"

	size = 100, 100
	im = Image.new("RGB", (width, height), font_color)
	txt = Image.new("RGB", (70, 70), font_color)
	draw = ImageDraw.Draw(txt)
	# 다람쥐
	unicode_font = ImageFont.truetype(os.path.join(Font_dir, font_name), font_size)
	draw.text((3, 0), unicode_text, font=unicode_font, fill=back_ground_color)
	txt = txt.rotate(17.5)
	im.paste(txt)

	# im.paste(txt)
	txt = Image.new("RGB", (70, 70), font_color)
	draw = ImageDraw.Draw(txt)
	unicode_font = ImageFont.truetype(os.path.join(Font_dir, font_name), font_size2)
	draw.text((3, 0), unicode_text2, font=unicode_font, fill=back_ground_color)
	txt = txt.rotate(17.5)
	im.paste(txt, (font_size,0))
	txt = Image.new("RGB", (70, 70), font_color)
	draw = ImageDraw.Draw(txt)
	unicode_font = ImageFont.truetype(os.path.join(Font_dir, font_name), font_size3)
	draw.text((3, 0), unicode_text3, font=unicode_font, fill=back_ground_color)
	txt = txt.rotate(17.5)
	im.paste(txt, (font_size + font_size2,0))
	#헌
	txt = Image.new("RGB", (70, 70), font_color)
	draw = ImageDraw.Draw(txt)
	unicode_font = ImageFont.truetype(os.path.join(Font_dir, font_name), font_size4)
	draw.text((3, 0), unicode_text4, font=unicode_font, fill=back_ground_color)
	txt = txt.rotate(17.5)
	im.paste(txt, (font_size + font_size2 + font_size3 + space_size,0))
	#쳇
	txt = Image.new("RGB", (70, 70), font_color)
	draw = ImageDraw.Draw(txt)
	unicode_font = ImageFont.truetype(os.path.join(Font_dir, font_name), font_size3)
	draw.text((3, 0), unicode_text5, font=unicode_font, fill=back_ground_color)
	txt = txt.rotate(17.5)
	im.paste(txt, (font_size + font_size2 + font_size3 + space_size + font_size4 + space_size2, 0))

	#바퀴에
	txt = Image.new("RGB", (70, 70), font_color)
	draw = ImageDraw.Draw(txt)
	unicode_font = ImageFont.truetype(os.path.join(Font_dir, font_name), font_size5)
	draw.text((3, 0), unicode_text6, font=unicode_font, fill=back_ground_color)
	txt = txt.rotate(17.5)
	im.paste(txt, (font_size + font_size2 + font_size3 + space_size + font_size4 + space_size2 + font_size3 + space_size2, 0))
	txt = Image.new("RGB", (70, 70), font_color)
	draw = ImageDraw.Draw(txt)
	unicode_font = ImageFont.truetype(os.path.join(Font_dir, font_name), font_size2)
	draw.text((3 , 0), unicode_text7, font=unicode_font, fill=back_ground_color)
	txt = txt.rotate(17.5)
	im.paste(txt, (font_size + font_size2 + font_size3 + space_size + font_size4 + space_size2 + font_size3 + space_size2 + font_size5, 0))
	txt = Image.new("RGB", (70, 70), font_color)
	draw = ImageDraw.Draw(txt)
	unicode_font = ImageFont.truetype(os.path.join(Font_dir, font_name), font_size3)
	draw.text((3 , 0), unicode_text8, font=unicode_font, fill=back_ground_color)
	txt = txt.rotate(17.5)
	im.paste(txt, (font_size + font_size2 + font_size3 + space_size + font_size4 + space_size2 + font_size3 + space_size2 + font_size5 + font_size2, 0))

	#타고파
	txt = Image.new("RGB", (70, 70), font_color)
	draw = ImageDraw.Draw(txt)
	unicode_font = ImageFont.truetype(os.path.join(Font_dir, font_name), font_size4)
	draw.text((3 , 0), unicode_text9, font=unicode_font, fill=back_ground_color)
	txt = txt.rotate(17.5)
	im.paste(txt, (font_size + font_size2 + font_size3 + space_size + font_size4 + space_size2 + font_size3 + space_size2 + font_size5 + font_size2 + font_size3 + space_size3, 0))
	txt = Image.new("RGB", (70, 70), font_color)
	draw = ImageDraw.Draw(txt)
	unicode_font = ImageFont.truetype(os.path.join(Font_dir, font_name), font_size2)
	draw.text((3, 0), unicode_text10, font=unicode_font, fill=back_ground_color)
	txt = txt.rotate(17.5)
	im.paste(txt, (font_size + font_size2 + font_size3 + space_size + font_size4 + space_size2 + font_size3 + space_size2 + font_size5 + font_size2 + font_size3 + space_size3+ font_size4, 0))
	txt = Image.new("RGB", (70, 70), font_color)
	draw = ImageDraw.Draw(txt)
	unicode_font = ImageFont.truetype(os.path.join(Font_dir, font_name), font_size3)
	draw.text((3, 0), unicode_text11, font=unicode_font, fill=back_ground_color)
	txt = txt.rotate(17.5)
	im.paste(txt, (font_size + font_size2 + font_size3 + space_size + font_size4 + space_size2 + font_size3 + space_size2 + font_size5 + font_size2 + font_size3 + space_size3 + font_size4 + font_size2 - 10, 0))
	ImageOps.invert(im).show()
	# im.show()
	# im.save(os.path.join('./' + font_name, 'base_line.jpg'))

	# Create_Font_maker(Font_dir, back_ground_color, font_color, font_name, font_size, korean_label)

def main():
	global Font_dir, korean_label

	# with open('../labels/256_common_hangul.txt', 'r', encoding='utf8') as f:
	#     for line in f:
	#         if 'str' in line:
	#             break
	#         korean_label.append(line[0])

	list_files = os.listdir(Font_dir)
	for i in list_files:
		# print(i)
		makeImage(i)
	# print(korean_label)


if __name__ == '__main__':
	main()

