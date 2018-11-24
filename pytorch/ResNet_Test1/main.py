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

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def main(model_dir, number):
    utils.default_model_dir = model_dir
    BATCH_SIZE = 128
    lr = 0.01
    EPOCH = 200 
    start_epoch = 0
    train_Data, test_Data = utils.Package_Data_onehot_Slice_Loder(number+1)
    
    train_loader = torch.utils.data.DataLoader(dataset=train_Data, batch_size=BATCH_SIZE, shuffle=True, num_workers = 4)
    test_loader = torch.utils.data.DataLoader(dataset=test_Data, batch_size=BATCH_SIZE, shuffle=False, num_workers = 4)

    utils.default_model_dir = model_dir
    
    start_time = time.time()

    model = ResNet()
    
    if torch.cuda.is_available():
        # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        print("USE", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model).cuda()
        cudnn.benchmark = True

    else:
        print("NO GPU -_-;")
        
    # Loss and Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion_MSE = nn.MSELoss().cuda()
    criterion_Cross_last = nn.CrossEntropyLoss().cuda()
    checkpoint = utils.load_checkpoint(model_dir+str(number))

    if not checkpoint:
        pass
    else:
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    for epoch in range(start_epoch, EPOCH+1):
        if epoch < 100:
            learning_rate = lr
        elif epoch < 150:
            learning_rate = lr * 0.1
        else:
            learning_rate = lr * 0.01
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate
    
        train(model, optimizer, criterion_MSE, criterion_Cross_last , train_loader, epoch)
        test(model, criterion_MSE, criterion_Cross_last , test_loader, epoch)
        utils.save_model_checkpoint(epoch, model, model_dir, number, optimizer)
        utils.check_model_result_image(epoch, model, number)
        
    # utils.conv_weight_L1_printing(model.module)
    now = time.gmtime(time.time() - start_time)
    print('{} hours {} mins {} secs for training'.format(now.tm_hour, now.tm_min, now.tm_sec))


# Train the model
def train(model, optimizer, criterion_MSE, criterion_Cross_last , train_loader, epoch):
    model.train()
    print_loss = 0
    print_loss2 = 0  
    total = 0
    correct = 0
    for batch_idx, (data, target, onehot_target) in enumerate(train_loader):
        if torch.cuda.is_available():
            data, target, onehot_target = Variable(data.cuda()), Variable(target.cuda()), Variable(onehot_target.cuda())
        else:
            data, target, onehot_target  = Variable(data), Variable(target), Variable(onehot_target)
            
        data, target, onehot_target = setting_data(data, target, onehot_target)
        optimizer.zero_grad()
        output = model(data)
        print(onehot_target.shape)
        print(output[0].shape)
        last_loss = criterion_Cross_last(output[0], onehot_target)
        image_loss = criterion_MSE(output[1], target)
        last_loss.backward(retain_graph=True)
        image_loss.backward(retain_graph=True)
        optimizer.step()
        print_loss += last_loss.item()
        print_loss2 += image_loss.item()
        _, predicted = torch.max(output[0].data, 1)
        total += target.size(0)
        correct += predicted.eq(onehot_target.data).sum()
        if batch_idx % 10 == 0:
            utils.print_log('Epoch: {} | Batch: {} |  Cross_Loss: ({:.4f}) | image_Loss: ({:.4f}) | Acc: ({:.2f}%) ({}/{})'
                  .format(epoch, batch_idx, print_loss / (batch_idx + 1), print_loss2 , 100. * correct / total, correct, total))
            print('Epoch: {} | Batch: {} |  Cross_Loss: ({:.4f}) | image_Loss: ({:.4f}) | Acc: ({:.2f}%) ({}/{})'
                  .format(epoch, batch_idx, print_loss / (batch_idx + 1), print_loss2 , 100. * correct / total, correct, total))
            
        
def test(model, criterion_MSE, criterion_Cross_last , test_loader, epoch):
    model.eval()
    print_loss = 0
    print_loss2 = 0  
    total = 0
    correct = 0
    for batch_idx, (data, target, onehot_target) in enumerate(test_loader):
        if torch.cuda.is_available():
            data, target, onehot_target = Variable(data.cuda()), Variable(target.cuda()), Variable(onehot_target.cuda())
        else:
            data, target, onehot_target = Variable(data), Variable(target), Variable(onehot_target)
            
        data, target, onehot_target = setting_data(data, target, onehot_target)
        output = model(data)
        print(onehot_target.shape)
        print(torch.max(onehot_target, 1)[1].shape)
        last_loss = criterion_Cross_last(output[0], torch.max(onehot_target, 1)[1])
        image_loss = criterion_MSE(output[1], target)
        
        print_loss += last_loss.item()
        print_loss2 += image_loss.item()
        _, predicted = torch.max(output[0].data, 1)
        total += target.size(0)
        correct += predicted.eq(onehot_target.data).sum()
        if batch_idx % 10 == 0:
            utils.print_log('# TEST : Epoch: {} | Batch: {} |  Cross_Loss: ({:.4f}) | image_Loss: ({:.4f}) | Acc: ({:.2f}%) ({}/{})'
                  .format(epoch, batch_idx, print_loss / (batch_idx + 1), print_loss2 , 100. * correct / total, correct, total))
            print('# TEST : Epoch: {} | Batch: {} |  Cross_Loss: ({:.4f}) | image_Loss: ({:.4f}) | Acc: ({:.2f}%) ({}/{})'
                  .format(epoch, batch_idx, print_loss / (batch_idx + 1), print_loss2 , 100. * correct / total, correct, total))


def setting_data(data, target, onehot_target):
    data = data.type(torch.cuda.FloatTensor)
    data = utils.renormalize_image(data)
    data = utils.normalize_function(data)
    target = target.type(torch.cuda.FloatTensor)
    target = utils.renormalize_image(target)
    target = utils.normalize_function(target)
    onehot_target = onehot_target.type(torch.cuda.LongTensor)
    onehot_target = onehot_target.squeeze()
    return data, target, onehot_target


def do_learning(model_dir, number):
    global max_result
    max_result = []
    main(model_dir, number)

if __name__ == '__main__':
    print(str(1)+'for train')	
    model_dir = '../ResNet_Test1/model/{}'.format(1)
    do_learning(model_dir, 1)
        
