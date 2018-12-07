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
from collections import OrderedDict

os.environ["CUDA_VISIBLE_DEVICES"] = '5'

def main(main_model_dir, number):
    
    utils.default_model_dir = main_model_dir + '/model/'
    BATCH_SIZE = 64
    lr = 0.001
    EPOCH = 20 
    start_epoch = 0
    start_time = time.time()

    train_Data, test_Data = utils.Package_Data_Slice_Loder(number)
    train_loader = torch.utils.data.DataLoader(dataset=train_Data, batch_size=BATCH_SIZE, shuffle=True, num_workers = 4)
    test_loader = torch.utils.data.DataLoader(dataset=test_Data, batch_size=BATCH_SIZE, shuffle=False, num_workers = 4)


    model =ResNet()


    if torch.cuda.is_available():
        print("USE", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model).cuda()
        cudnn.benchmark = True
    else:
        print("NO GPU -_-;")
    

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # criterion_Cross = nn.CrossEntropyLoss().cuda()
    criterion_MSE = nn.MSELoss().cuda()


    checkpoint = utils.load_checkpoint(utils.default_model_dir)
    if not checkpoint:
        pass
    else:
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])


    # start train    
    for epoch in range(start_epoch, EPOCH+1):
        if epoch < 6:
            learning_rate = lr
        elif epoch < 13:
            learning_rate = lr * 0.1
        else:
            learning_rate = lr * 0.01
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate
    
        train(model, optimizer, criterion_MSE, train_loader, epoch)
        test(model, criterion_MSE, test_loader, epoch)
        utils.save_model_checkpoint(epoch, model, utils.default_model_dir, optimizer)
        utils.check_model_result_image(epoch, model, number, main_model_dir)

    now = time.gmtime(time.time() - start_time)
    print('{} hours {} mins {} secs for training'.format(now.tm_hour, now.tm_min, now.tm_sec))


# Train the model
def train(model, optimizer, criterion_MSE, train_loader, epoch):
    model.train()
    print_loss = 0
    total = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if torch.cuda.is_available():
            data, target = Variable(data.cuda()), Variable(target.cuda())
        else:
            data, target = Variable(data), Variable(target)
            
        data, target = setting_data(data, target)
        optimizer.zero_grad()
        output = model(data)
        image_loss = criterion_MSE(output, target)
        # Cross_Loss.backward(retain_graph=True)
        # image_loss.backward(retain_graph=True)
        image_loss.backward()
        optimizer.step()
        print_loss += image_loss.item()
        if batch_idx % 10 == 0:
            utils.print_log('Epoch: {} | Batch: {}  | image_Loss: ({:.4f}) | Best label epoch : {} | Best Acc: ({:.2f}%)'
                  .format(epoch, batch_idx, print_loss, utils.cross_epoch, utils.cross_correct ))
            print('Epoch: {} | Batch: {}  | image_Loss: ({:.4f}) | Best label epoch : {} | Best Acc: ({:.2f}%)'
                  .format(epoch, batch_idx, print_loss, utils.cross_epoch, utils.cross_correct ))
            
        
def test(model, criterion_MSE, test_loader, epoch):
    model.eval()
    print_loss = 0
    total = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        if torch.cuda.is_available():
            data, target = Variable(data.cuda()), Variable(target.cuda())
        else:
            data, target = Variable(data), Variable(target)
            
        data, target = setting_data(data, target)
        output = model(data)
        image_loss = criterion_MSE(output, target)
        print_loss += image_loss.item()
        if batch_idx % 10 == 0:
            utils.print_log('Test# Epoch: {} | Batch: {}  | image_Loss: ({:.4f}) | Best label epoch : {} | Best Acc: ({:.2f}%)'
                  .format(epoch, batch_idx, print_loss, utils.cross_epoch, utils.cross_correct ))
            print('Test# Epoch: {} | Batch: {}  | image_Loss: ({:.4f}) | Best label epoch : {} | Best Acc: ({:.2f}%)'
                  .format(epoch, batch_idx, print_loss, utils.cross_epoch, utils.cross_correct ))



def setting_data(data, target):
    data = data.type(torch.cuda.FloatTensor)
    target = target.type(torch.cuda.FloatTensor)
    return data, target

def do_learning(main_model_dir, korean_model_dir, number):
    main(main_model_dir, korean_model_dir, number)

if __name__ == '__main__':
    
    dataset_num = 1

    print('Dataset numper is {}'.format(dataset_num))

    main_model_dir = '/data2/hhjung/Sonmat_Result/Resnet_Fifth' 

    do_learning(main_model_dir, dataset_num)
        
