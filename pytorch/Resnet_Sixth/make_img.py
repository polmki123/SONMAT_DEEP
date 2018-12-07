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

def main(model_dir, number):
    utils.default_model_dir = model_dir + '/model/'
    epoch = 'result'
    start_time = time.time()

    model = ResNet()
    
    if torch.cuda.is_available():
        # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        print("USE", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model).cuda()
        cudnn.benchmark = True

    else:
        print("NO GPU -_-;")

    checkpoint = utils.load_checkpoint(utils.default_model_dir)

    if not checkpoint:
        pass
    else:
        model.load_state_dict(checkpoint['state_dict'])
        
    utils.check_model_result_image(epoch, model, number, model_dir)   
        
    # utils.conv_weight_L1_printing(model.module)
    now = time.gmtime(time.time() - start_time)
    print('{} hours {} mins {} secs for training'.format(now.tm_hour, now.tm_min, now.tm_sec))


def do_learning(model_dir, number):
    global max_result
    max_result = []
    main(model_dir, number)

if __name__ == '__main__':
    print(str(1)+'for train')	
    model_dir = '/data2/hhjung/Sonmat_Result/Resnet_Sixth' 
    do_learning(model_dir, 1)
        
