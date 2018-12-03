import torch

import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import utils
import os

import glob
from PIL import Image
import numpy as np
import PIL.ImageOps

import argparse

import numpy as np
import math
import itertools
import time
import datetime
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from models import *
from datasets import *


import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"] = '7'

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--dataset_name', type=str, default="facades", help='name of the dataset')
parser.add_argument('--batch_size', type=int, default=128, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch from which to start lr decay')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_height', type=int, default=64, help='size of image height')
parser.add_argument('--img_width', type=int, default=64, help='size of image width')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=500, help='interval between sampling of images from generators')
parser.add_argument('--checkpoint_interval', type=int, default=-1, help='interval between model checkpoints')
opt = parser.parse_args()

cuda = True if torch.cuda.is_available() else False

# Loss functions
criterion_GAN = torch.nn.MSELoss()
criterion_pixelwise = torch.nn.L1Loss()

# Loss weight of L1 pixel-wise loss between translated image and real image
lambda_pixel = 100

# Calculate output of image discriminator (PatchGAN)
patch = (1, opt.img_height//2**4, opt.img_width//2**4)


# Initialize generator and discriminator
generator = GeneratorUNet()
discriminator = Discriminator()

if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    criterion_GAN.cuda()
    criterion_pixelwise.cuda()

generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)


# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))


train_loader = torch.utils.data.DataLoader(dataset=train_Data, batch_size=128, shuffle=True, num_workers = 4)
test_loader = torch.utils.data.DataLoader(dataset=test_Data, batch_size=128, shuffle=False, num_workers = 4)

def to_variable(x):
    if torch.cuda.is_available:
        x = x.cuda()
    return Variable(x)


def main(model_dir, number):
    Discri = Discriminator()
    Gener = Generator()
    if torch.cuda.is_available():
        # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        print("USE", torch.cuda.device_count(), "GPUs!")
        Discri = tnn.DataParallel(Discri).cuda()
        Gener = tnn.DataParallel(Gener).cuda()
        cudnn.benchmark = True

    else:
        print("NO GPU -_-;")

    Discri.apply(weights_init)
    Gener.apply(weights_init)

    utils.default_model_dir = model_dir

    start_time = time.time()
    
    # Loss and Optimizer
    Gener_checkpoint = utils.Gener_load_checkpoint(model_dir+str(1))
    Discri_checkpoint = utils.Discri_load_checkpoint(model_dir+str(1))

    if not Gener_checkpoint:
        pass
    else:
        #start_epoch = checkpoint['epoch'] + 1
        Gener.load_state_dict(Gener_checkpoint['state_dict'])
        #optimizerG.load_state_dict(Gener_checkpoint['optimizer'])

    if not Discri_checkpoint:
        pass
    else:
        #start_epoch = checkpoint['epoch'] + 1
        Discri.load_state_dict(Discri_checkpoint['state_dict'])
        #optimizerD.load_state_dict(Discri_checkpoint['optimizer'])
    
    saveimagedir = '../GAN_model/save_font_image/' + str(number) + '/' + str(1000) + '/'
    inputimagedir = '../GAN_model/test1.jpg'
    input_data = input_Deepmodel_image(inputimagedir)
    Gener.eval()
    check_point = 0
    for i in input_data:
        check_point = check_point + 1
        i = np.array(i)
        i = i.reshape(1,9,64,64)
        input = torch.from_numpy(i)
        input = normalize_function(input)
        input = input.type(torch.cuda.FloatTensor)
        noise3 = torch.randn(1, 1, 64, 64).uniform_(0,1)
        noise3 = noise3.type(torch.cuda.FloatTensor)
        Data_noise = (torch.cat([input,noise3],dim=1))
        Data_noise = Data_noise.type(torch.cuda.FloatTensor)
        output = Gener(Data_noise)
        output = Variable(output).data.cpu().numpy()
        output = output.reshape(64,64)
        #print(output)
        output = (output)*255
        img = Image.fromarray(output.astype('uint8'), 'L')
        #img = PIL.ImageOps.invert(img)
        if not os.path.exists(saveimagedir):
            os.makedirs(saveimagedir)
        img.save(saveimagedir + str(check_point) + 'my.jpg')

def renormalize_image(img):
    renormalized = (img + 1) * 127.5
    return renormalized

def normalize_function(img):
    img = (img - img.min()) / (img.max() - img.min())
    #img = (img - img.mean()) / (img.std())
    return img

def input_Deepmodel_image(inputimagedir) :
    frame_dir = '../Deep_model/frame_label/'
    frame_paths = glob.glob(os.path.join(frame_dir, '*.jpg'))
    input_data = list()
    for frame in frame_paths :
        frame_image = np.array(Image.open(frame)).reshape(1,64,64)
        input_image = np.array(Image.open(inputimagedir))
        input_image = np.array(np.split(input_image, 8, axis=1))  # 8*64*64
        Concat_data = np.append(input_image, frame_image, axis=0)
        if ((9, 64, 64) == Concat_data.shape):
            input_data.append(Concat_data)
            
    return input_data

def make_Gan_image(model_dir, number):
    main(model_dir, number)

if __name__ == '__main__':
    print('make image')
    model_dir = '../GAN_model/model/'
    make_Gan_image(model_dir, 1)
        


 




