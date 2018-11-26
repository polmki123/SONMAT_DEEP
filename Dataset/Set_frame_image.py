import os
import glob

unicode_file = open("../labels/2350_common_hangul_unicode_ANSI.txt", 'r')
image_list = os.listdir("../labels/frame_label/")
unicode_path = '../labels/2350_common_hangul_unicode_ANSI.txt'
image_path = '../labels/frame_label/'
lines = unicode_file.readlines()
i=0

for f in image_list:
    new_name = lines[i]
    os.rename(image_path+f,new_name[:-1]+str('.jpg'))
    i+=1

unicode_file.close()