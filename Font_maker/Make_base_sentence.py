from PIL import Image, ImageDraw, ImageFont, ImageFilter
import os

korean_label = []
Font_dir = '../Font_maker/Font/'


def makeImage(font_name):
	global Font_dir, korean_label
	if not os.path.exists(font_name):
		os.makedirs(font_name)
	
	width=460
	height=64
	back_ground_color=(255,255,255)
	font_color=(0,0,0)
	font_size=36

	unicode_text = u"다람쥐 헌 쳇바퀴에 타고파"

	im  =  Image.new ( "RGB", (width,height), back_ground_color )
	draw  =  ImageDraw.Draw ( im )
	unicode_font = ImageFont.truetype(os.path.join(Font_dir, font_name), font_size)
	draw.text ( (10,10), unicode_text, font=unicode_font, fill=font_color )

	im.save(os.path.join(font_name + '.jpg'))


	for label_item in korean_label:

		label_width=25
		label_height=37

		im  =  Image.new ( "RGB", (label_width,label_height), back_ground_color )
		draw  =  ImageDraw.Draw ( im )
		unicode_font = ImageFont.truetype(os.path.join(Font_dir, font_name), font_size)
		draw.text ( (0,0), label_item, font=unicode_font, fill=font_color )

		im.save(os.path.join('./' + font_name, label_item + '.jpg'))

def main():
	global Font_dir, korean_label

	list_files = os.listdir( Font_dir )
	for i in list_files:
		makeImage(i)

if __name__=='__main__':
	main()

