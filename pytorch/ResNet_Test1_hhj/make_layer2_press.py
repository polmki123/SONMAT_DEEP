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
import gzip
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = '6'

def main(model_dir, number,package_dir):
    utils.default_model_dir = model_dir
    BATCH_SIZE = 128
    lr = 0.0002
    EPOCH = 200
    start_epoch = 0
    train_Data, test_Data = utils.Package_Data_onehot_Slice_Loder(number+1)
    
    train_loader = torch.utils.data.DataLoader(dataset=train_Data, batch_size=1, shuffle=False, num_workers = 4)
    test_loader = torch.utils.data.DataLoader(dataset=test_Data, batch_size=1, shuffle=False, num_workers = 4)

    utils.default_model_dir = model_dir
    
    start_time = time.time()

    model = ResNet()
    

    if torch.cuda.is_available():
        # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        print("USE", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model).cuda()
        # cudnn.benchmark = True

    else:
        print("NO GPU -_-;")
        
    # Loss and Optimizer
    checkpoint = utils.load_checkpoint(model_dir)

    if not checkpoint:
        pass
    else:
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['state_dict'])

    
    train(model, train_loader,package_dir)
    test(model, test_loader,package_dir)
    
    
    # utils.conv_weight_L1_printing(model.module)
    now = time.gmtime(time.time() - start_time)
    print('{} hours {} mins {} secs for training'.format(now.tm_hour, now.tm_min, now.tm_sec))


# Train the model
def train(model, train_loader,package_dir):
    model.eval()
    with gzip.open(package_dir + 'Resnet_train.pkl', 'wb') as ft:
        X_datas = []
        label_datas =[]
        for batch_idx, (data, target, onehot_target) in enumerate(train_loader):
            if torch.cuda.is_available():
                data, target, onehot_target = Variable(data.cuda()), Variable(target.cuda()), Variable(onehot_target.cuda())
            else:
                data, target, onehot_target  = Variable(data), Variable(target), Variable(onehot_target)
                
            data, target, onehot_target = setting_data(data, target, onehot_target)
            frame_data = data[0][8]
            frame_data = Variable(frame_data).data.cpu().numpy()
            frame_data = frame_data.reshape(1,64,64)
            output = model(data)
            output = Variable(output[1]).data.cpu().numpy()
            output = output.reshape(64,64)
            output = utils.renormalize_image(output)
            output = utils.normalize_function(output)
            img = Image.fromarray(output.astype('uint8'), 'L')
            img = PIL.ImageOps.invert(img)
            output = utils.normalize_image(np.array(img))
            output = output.reshape(1,64,64)
            target = Variable(target).data.cpu().numpy()
            target = target.reshape(1,64,64)
            Concat_data = np.append(output, frame_data, axis=0)
            X_datas.append(Concat_data)
            label_datas.append(target)
            if len(X_datas) == 2350 :
                X_datas = np.array(X_datas)
                print(X_datas.shape)
                label_datas = np.array(label_datas)
                print(label_datas.shape)
                example = (X_datas, label_datas)
                pickle.dump(example, ft)
                X_datas = []
                label_datas = []
            

def test(model, train_loader,package_dir):
    model.eval()
    with gzip.open(package_dir + 'Resnet_test.pkl', 'wb') as ft:
        X_datas = []
        label_datas =[]
        for batch_idx, (data, target, onehot_target) in enumerate(train_loader):
            if torch.cuda.is_available():
                data, target, onehot_target = Variable(data.cuda()), Variable(target.cuda()), Variable(onehot_target.cuda())
            else:
                data, target, onehot_target  = Variable(data), Variable(target), Variable(onehot_target) 
               
            data, target, onehot_target = setting_data(data, target, onehot_target) 
            frame_data = data[0][8]
            frame_data = Variable(frame_data).data.cpu().numpy()
            frame_data = frame_data.reshape(1,64,64)
            output = model(data)
            output = Variable(output[1]).data.cpu().numpy()
            output = output.reshape(64,64)
            output = utils.renormalize_image(output)
            output = utils.normalize_function(output)
            img = Image.fromarray(output.astype('uint8'), 'L')
            img = PIL.ImageOps.invert(img)
            output = utils.normalize_image(np.array(img))
            output = output.reshape(1,64,64)
            target = Variable(target).data.cpu().numpy()
            target = target.reshape(1,64,64)
            Concat_data = np.append(output, frame_data, axis=0)
            X_datas.append(Concat_data)
            label_datas.append(target)
            if len(X_datas) == 2350 :
                X_datas = np.array(X_datas)
                print(X_datas.shape)
                label_datas = np.array(label_datas)
                print(label_datas.shape)
                example = (X_datas, label_datas)
                pickle.dump(example, ft)
                X_datas = []
                label_datas = []
        

def setting_data(data, target, onehot_target):
    data = data.type(torch.cuda.FloatTensor)
    target = target.type(torch.cuda.FloatTensor)
    onehot_target = onehot_target.type(torch.cuda.LongTensor)
    onehot_target = torch.squeeze(onehot_target)
    return data, target, onehot_target


def do_learning(model_dir, number,package_dir):
    global max_result
    max_result = []
    main(model_dir, number,package_dir)

if __name__ == '__main__':
    print(str(1)+'for train')
    package_dir = '../ResNet_Test1/Conpress/'
    model_dir = '../ResNet_Test1/model/'
    do_learning(model_dir, 1,package_dir)
    
