import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
#from . import utils
import utils
import os
import time
import glob
from PIL import Image
import numpy as np
import PIL.ImageOps
from model import *

os.environ["CUDA_VISIBLE_DEVICES"] = '6'

def input_Deepmodel_image(inputimagedir):
	frame_dir = '../Deep_model/frame_label/'
	frame_paths = glob.glob(os.path.join(frame_dir, '*.jpg'))
	input_data = list()
	for frame in frame_paths:
		frame_image = np.array(Image.open(frame)).reshape(1, 64, 64)
		input_image = np.array(Image.open(inputimagedir))
		input_image = np.array(np.split(input_image, 8, axis=1))  # 8*64*64
		Concat_data = np.append(input_image, frame_image, axis=0)
		if ((9, 64, 64) == Concat_data.shape):
			input_data.append(Concat_data)

	return input_data



def main(inputimagedir, model_dir):
	start_time = time.time()
	input_data = utils.input_Deepmodel_image(inputimagedir)
	utils.default_model_dir = model_dir

	model = ResNet()
	model = nn.DataParallel(model)
	# if torch.cuda.is_available():
	#    print("USE", torch.cuda.device_count(), "GPUs!")
	#    model = tnn.DataParallel(model).cuda()
	#    cudnn.benchmark = True
	# else:
	#    print("NO GPU -_-;")

	checkpoint = utils.load_checkpoint(model_dir+str(12))
	if not checkpoint:
		pass
	else:
		model.load_state_dict(checkpoint['state_dict'])
		model.eval()
		number = 0
		for i in input_data:
			number = number + 1
			i = np.array(i)
			i = i.reshape(1,9,64,64)
			input = torch.from_numpy(i)
			# input = Variable(input)
			input = input.type(torch.FloatTensor)
			input = utils.normalize_image(input)
			output = model(input)
			output = Variable(output[1]).data.cpu().numpy()
			# print(output.shape)
			output = output.reshape(64,64)
			output = utils.renormalize_image(output)
			output = utils.normalize_function(output)
			img = Image.fromarray(output.astype('uint8'), 'L')
			img = PIL.ImageOps.invert(img)
			img.save('../ResNet_Test1/save_image/' + str(number) + 'my.png')

        
	now = time.gmtime(time.time() - start_time)
	print('{} hours {} mins {} secs for data'.format(now.tm_hour, now.tm_min, now.tm_sec))
if __name__ == "__main__":
	inputimagedir = '../Deep_model/test1.jpg'
	model_dir = '../ResNet_Test1/model/'
	main(inputimagedir, model_dir)
