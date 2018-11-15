from PIL import Image, ImageDraw, ImageFont, ImageFilter
import os

korean_label = []
Font_dir = '../Font/'


def makeImage(font_name):
    global Font_dir, korean_label
    if not os.path.exists(font_name):
        os.makedirs(font_name)

    #configuration
    width=512
    height=64
    back_ground_color=(255,255,255)
    font_size=36
    font_color=(0,0,0)

    for label_item in korean_label:

        label_width=64
        label_height=64

        im  =  Image.new ( "RGB", (label_width,label_height), back_ground_color )
        draw  =  ImageDraw.Draw ( im )
        unicode_font = ImageFont.truetype(os.path.join(Font_dir, font_name), font_size)
        draw.text ( (15,11), label_item, font=unicode_font, fill=font_color )

        im.save(os.path.join('./' + font_name, label_item + '.jpg'))

def main():
    global Font_dir, korean_label
    
    with open('../labels/2350_common_hangul.txt', 'r', encoding='utf8') as f:
        for line in f:
            if 'str' in line:
                break
            korean_label.append(line[0])
    
    list_files = os.listdir(Font_dir)
    for i in list_files:
        makeImage(i)
    print(korean_label)

if __name__=='__main__':
    main()

