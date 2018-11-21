import torch
import torch.nn as tnn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
import utils
import os
import time
import vgg16
import glob
import os
import pickle
from PIL import Image
import numpy as np
from collections import OrderedDict
import PIL.ImageOps
class VGG16(tnn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.layer1 = tnn.Sequential(
            
            # 1-1 conv layer
            # batch_size * 3*64*64
            tnn.Conv2d(9, 64, kernel_size=3, padding=1),
            tnn.BatchNorm2d(64),
            tnn.ReLU(),
            
            # 1-2 conv layer
            # batch_size * 64*64*64
            tnn.Conv2d(64, 64, kernel_size=3, padding=1),
            tnn.BatchNorm2d(64),
            tnn.ReLU(),
            
            # 1 Pooling layer
            # batch_size * 64*64*64
            tnn.MaxPool2d(kernel_size=2, stride=2))
        
        self.layer2 = tnn.Sequential(
            
            # 2-1 conv layer
            # batch_size * 64*32*32
            tnn.Conv2d(64, 128, kernel_size=3, padding=1),
            tnn.BatchNorm2d(128),
            tnn.ReLU(),
            
            # 2-2 conv layer
            # batch_size * 128*32*32
            tnn.Conv2d(128, 128, kernel_size=3, padding=1),
            tnn.BatchNorm2d(128),
            tnn.ReLU(),
            
            # 2 Pooling lyaer
            # batch_size * 128*32*32
            tnn.MaxPool2d(kernel_size=2, stride=2))
        
        self.layer3 = tnn.Sequential(
            
            # 3-1 conv layer
            # batch_size * 128*16*16
            tnn.Conv2d(128, 256, kernel_size=3, padding=1),
            tnn.BatchNorm2d(256),
            tnn.ReLU(),
            
            # 3-2 conv layer
            # batch_size * 256*16*16
            tnn.Conv2d(256, 256, kernel_size=3, padding=1),
            tnn.BatchNorm2d(256),
            tnn.ReLU(),
            
            # 3 Pooling layer
            # batch_size * 256*16*16
            tnn.MaxPool2d(kernel_size=2, stride=2))
        
        self.layer4 = tnn.Sequential(
            
            # 4-1 conv layer
            # batch_size * 512*8*8
            tnn.Conv2d(256, 512, kernel_size=3, padding=1),
            tnn.BatchNorm2d(512),
            tnn.ReLU(),
            
            # 4-2 conv layer
            # batch_size * 512*8*8
            tnn.Conv2d(512, 512, kernel_size=3, padding=1),
            tnn.BatchNorm2d(512),
            tnn.ReLU(),
            
            # 4 Pooling layer
            # batch_size * 512*8*8
            tnn.MaxPool2d(kernel_size=2, stride=2))
        
        self.layer5 = tnn.Sequential(
            
            # 5-1 conv layer
            # batch_size * 512*4*4
            tnn.Conv2d(512, 512, kernel_size=3, padding=1),
            tnn.BatchNorm2d(512),
            tnn.ReLU(),
            
            # 5-2 conv layer
            # batch_size * 512*4*4
            tnn.Conv2d(512, 256, kernel_size=3, padding=1),
            tnn.BatchNorm2d(256),
            tnn.ReLU())
            
            # 5 Pooling layer
            # batch_size * 4*4*512
            # tnn.MaxPool2d(kernel_size=2, stride=2))
        
        self.layer6 = tnn.Sequential(
            
            # 6 Transpose
            # batch_size * 256*4*4
            tnn.ConvTranspose2d(256, 64, kernel_size = 4, stride = 4, padding = 0),
            tnn.BatchNorm2d(64),
            tnn.ReLU())
        
        self.layer7 = tnn.Sequential(
            
            # 7 Transpose
            # batch_size * 64*16*16
            tnn.ConvTranspose2d(64, 16, kernel_size=4, stride=2, padding=1),
            tnn.BatchNorm2d(16),
            tnn.ReLU())
        
        self.layer8 = tnn.Sequential(
            
            # 8 Transpose
            # batch_size * 16*32*32
            tnn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
            tnn.Tanh())
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

def normalize_image(img):
    """
    Make image zero centered and in between (-1, 1)
    """
    normalized = (img / 127.5) - 1.
    return normalized

def renormalize_image(img):
    renormalized = (img + 1) * 127.5
    return renormalized
def normalize_function(img):
    img = (img - img.min()) / (img.max() - img.min())
    img = (img - img.mean()) / (img.std())
    return img
def input_Deepmodel_imge(inputimagedir) :
    frame_dir = '../Deep_model/frame_label/'
    frame_paths = glob.glob(os.path.join(frame_dir, '*.jpg'))
    input_data = list()
    for frame in frame_paths :
        frame_image = np.array(Image.open(frame)).reshape(1,64,64)
        #frame_image = normalize_function(frame_image)
        #frame_image = frame_image
        #print(frame_image.shape)
        input_image = np.array(Image.open(inputimagedir))
        input_image = np.array(np.split(input_image, 8, axis=1))  # 8*64*64
        #input_image = input_image/255
        #input_image = normalize_function(input_image)
        Concat_data = np.append(input_image, frame_image, axis=0)
        #Concat_data = normalize_function(Concat_data)
        if ((9, 64, 64) == Concat_data.shape):
            input_data.append(Concat_data)
        #input_data = normalize_image(input_data)
    return input_data


def main(inputimagedir, model_dir):
    start_time = time.time()
    input_data = input_Deepmodel_imge(inputimagedir)
    utils.default_model_dir = model_dir
    model = VGG16()
    if torch.cuda.is_available():
       # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
       print("USE", torch.cuda.device_count(), "GPUs!")
       model = tnn.DataParallel(model).cuda()
       cudnn.benchmark = True
    else:
       print("NO GPU -_-;")

    checkpoint = utils.load_checkpoint(model_dir+str(1))

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
        #input = input/255
        input = Variable(input.cuda())
        input = input.type(torch.cuda.FloatTensor)
        input = normalize_function(input)
        output = model(input)
        output = Variable(output).data.cpu().numpy()
        output = output.reshape(64,64)
        #output = 255*(output)
        img = Image.fromarray(output.astype('uint8'), 'L')
        img = PIL.ImageOps.invert(img)
        img.save('../Deep_model/save_image/' + str(number) + 'my.jpg')
        
    now = time.gmtime(time.time() - start_time)
    print('{} hours {} mins {} secs for data'.format(now.tm_hour, now.tm_min, now.tm_sec))
if __name__ == "__main__":
    inputimagedir = '../Deep_model/test1.jpg'
    model_dir = '../Deep_model/model/'
    main(inputimagedir, model_dir)
