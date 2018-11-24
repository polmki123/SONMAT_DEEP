import torch
import torch.nn as tnn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
import utils
import os
import time
import glob
from PIL import Image
import numpy as np
import PIL.ImageOps

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


class Generator(tnn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.layer1 = tnn.Sequential(
            # 1-1 conv layer
            # batch_size * 3*64*64
            tnn.Conv2d(10, 64, kernel_size=3, padding=1),
            tnn.BatchNorm2d(64),
            tnn.LeakyReLU(0.1),
            
            # 1-2 conv layer
            # batch_size * 64*64*64
            tnn.Conv2d(64, 64, kernel_size=3, padding=1),
            tnn.BatchNorm2d(64),
            tnn.LeakyReLU(0.1),
            
            # 1 Pooling layer
            # batch_size * 64*64*64
            tnn.MaxPool2d(kernel_size=2, stride=2))
        
        self.layer2 = tnn.Sequential(
            
            # 2-1 conv layer
            # batch_size * 64*32*32
            tnn.Conv2d(64, 128, kernel_size=3, padding=1),
            tnn.BatchNorm2d(128),
            tnn.LeakyReLU(0.1),
            
            # 2-2 conv layer
            # batch_size * 128*32*32
            tnn.Conv2d(128, 128, kernel_size=3, padding=1),
            tnn.BatchNorm2d(128),
            tnn.LeakyReLU(0.1),
            
            # 2 Pooling lyaer
            # batch_size * 128*32*32
            tnn.MaxPool2d(kernel_size=2, stride=2))
        
        self.layer3 = tnn.Sequential(
            
            # 3-1 conv layer
            # batch_size * 128*16*16
            tnn.Conv2d(128, 256, kernel_size=3, padding=1),
            tnn.BatchNorm2d(256),
            tnn.LeakyReLU(0.1),
            
            # 3-2 conv layer
            # batch_size * 256*16*16
            tnn.Conv2d(256, 256, kernel_size=3, padding=1),
            tnn.BatchNorm2d(256),
            tnn.LeakyReLU(0.1),
            
            # 3 Pooling layer
            # batch_size * 256*16*16
            tnn.MaxPool2d(kernel_size=2, stride=2))
        
        self.layer4 = tnn.Sequential(
            
            # 4-1 conv layer
            # batch_size * 512*8*8
            tnn.Conv2d(256, 512, kernel_size=3, padding=1),
            tnn.BatchNorm2d(512),
            tnn.LeakyReLU(0.1),
            
            # 4-2 conv layer
            # batch_size * 512*8*8
            tnn.Conv2d(512, 512, kernel_size=8,stride = 1, padding=0),
            tnn.BatchNorm2d(512),
            tnn.LeakyReLU(0.1),
            
            # 4 Pooling layer
            # batch_size * (64 * 8) * 1 * 1
            tnn.ConvTranspose2d(64 * 8, 64 * 16, 4, 1, 0))
        
        self.layer5 = tnn.Sequential(
            
            # 5-1 conv layer
            # batch_size * (64*16)*4*4
            tnn.ConvTranspose2d(64 * 16, 64 * 8, 4, 2, 1),
            tnn.BatchNorm2d(512),
            tnn.LeakyReLU(0.1),
            
            # 5-2 conv layer
            # batch_size * 512*8*8
            tnn.ConvTranspose2d(64 * 8, 64 * 2, 4, 2, 1),
            tnn.BatchNorm2d(128),
            tnn.LeakyReLU(0.1))
            
            # 5 Pooling layer
            # batch_size * 4*4*512
            # tnn.MaxPool2d(kernel_size=2, stride=2))
        
        self.layer6 = tnn.Sequential(
            
            # 6 Transpose
            # batch_size * (64 * 2) * 16 * 16
            tnn.ConvTranspose2d(64 * 2, 32, 4, 2, 1),
            tnn.BatchNorm2d(32),
            tnn.LeakyReLU(0.1)
            )
        
        self.layer7 = tnn.Sequential(
            
            # 7 Transpose
            # batch_size * 32*32*32
            tnn.ConvTranspose2d(32, 8, kernel_size=4, stride=2, padding=1),
            tnn.BatchNorm2d(8),
            tnn.LeakyReLU(0.1))
        
        self.layer8 = tnn.Sequential(
            
            # 8 Transpose
            # batch_size * 8*64*64
            tnn.Conv2d(8, 1, kernel_size=3, padding=1),
            tnn.LeakyReLU(0.1))
            # batch_size * 1*64*64
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
            
        return out

class Discriminator(tnn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = tnn.Sequential(
            #""" color imgage (fake or real image)"""
            # 1 * 64 * 64
            tnn.Conv2d(1,64,kernel_size = 4, stride = 2, padding = 1, bias = False),
            tnn.LeakyReLU(0.2, inplace = True),
            
            # 64 * 32 * 32
            tnn.Conv2d(64,128,kernel_size = 4, stride = 2, padding = 1, bias = False),
            tnn.BatchNorm2d(128),
            tnn.LeakyReLU(0.2, inplace = True),
            
            # 128 * 16 * 16
            tnn.Conv2d(128, 256, kernel_size = 4, stride = 2, padding = 1, bias = False),
            tnn.BatchNorm2d(256),
            tnn.LeakyReLU(0.2, inplace = True),
            
            # 256 * 8 * 8
            tnn.Conv2d(256, 512, kernel_size = 4, stride = 2, padding = 1, bias = False),
            tnn.BatchNorm2d(512),
            tnn.LeakyReLU(0.2, inplace = True),
            )
        
        # 512 * 4 * 4
        self.fc = tnn.Sequential(
            tnn.Linear(512*4*4 , 512),
            tnn.Linear(512, 256),
            tnn.Linear(256, 128),
            tnn.Sigmoid()
        )
        
    def forward(self, input, b_size):
        output = self.main(input)
        output = self.fc(output.view(b_size,-1))
        return output

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:         # Conv weight init
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:  # BatchNorm weight init
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

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
        


 




