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
    BATCH_SIZE = 128
    LEARNING_RATE = 0.01
    EPOCH = 200 
    
    train_Data, test_Data = utils.Package_Data_Slice_Loder(number+1)
    
    train_loader = torch.utils.data.DataLoader(dataset=train_Data, batch_size=BATCH_SIZE, shuffle=True, num_workers = 4)
    test_loader = torch.utils.data.DataLoader(dataset=test_Data, batch_size=BATCH_SIZE, shuffle=False, num_workers = 4)

    utils.default_model_dir = model_dir
    lr = 0.01
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
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion_MSE = nn.MSELoss().cuda()
    #criterion_Cross_middle = nn.CrossEntropyLoss().cuda()
    criterion_Cross_last = nn.CrossEntropyLoss().cuda()
    #criterion = nn.BCELoss().cuda()
    start_epoch = 0
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
    
        train(model, optimizer, criterion_MSE, criterion_Cross_middle, criterion_Cross_last , train_loader, epoch)
        test(model, optimizer, criterion_MSE, criterion_Cross_middle, criterion_Cross_last , train_loader, epoch)
        utils.save_model_checkpoint(epoch, model, model_dir, number, optimizer)
        utils.check_model_result_image(epoch, model, number)
        
    # utils.conv_weight_L1_printing(model.module)
    now = time.gmtime(time.time() - start_time)
    print('{} hours {} mins {} secs for training'.format(now.tm_hour, now.tm_min, now.tm_sec))


# Train the model
def train(model, optimizer, criterion_MSE, criterion_Cross_middle, criterion_Cross_last , train_loader, epoch):
    model.train()
    train_loss = 0
    total = 0
    correct = 0
    for batch_idx, (data, target, onehot_target) in enumerate(train_loader):
        if torch.cuda.is_available():
            data, target = Variable(data.cuda()), Variable(target.cuda()), Variable(onehot_target.cuda())
        else:
            data, target = Variable(data), Variable(target), Variable(onehot_target)
            
        data, target, onehot_target = setting_data(data, target, onehot_target)
        optimizer.zero_grad()
        output = model(data)
        image_loss = criterion_MSE(output[2], target)
        middle_loss = criterion_Cross_middle(output[1], onehot_target)
        last_loss = criterion_Cross_last(output[0], onehot_target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        #correct += predicted.eq(target.data).sum()
        if batch_idx % 10 == 0:
            utils.print_log('Epoch: {} | Batch: {} |  Loss: ({:.4f}) | Acc: ({:.2f}%) ({}/{})'
                  .format(epoch, batch_idx, train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
            print('Epoch: {} | Batch: {} |  Loss: ({:.4f}) | Acc: ({:.2f}%) ({}/{})'
                  .format(epoch, batch_idx, train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
            
        
def test(model, optimizer, criterion_MSE, criterion_Cross_middle, criterion_Cross_last , train_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (data, target, onehot_target) in enumerate(train_loader):
        if torch.cuda.is_available():
            data, target = Variable(data.cuda()), Variable(target.cuda()), Variable(onehot_target.cuda())
        else:
            data, target = Variable(data), Variable(target), Variable(onehot_target)
        data, target, onehot_target = setting_data(data, target, onehot_target)
        data, target = setting_data(data, target)
        output = model(data)
        image_loss = criterion_MSE(output[2], target)
        middle_loss = criterion_Cross_middle(output[1], onehot_target)
        last_loss = criterion_Cross_last(output[0], onehot_target)
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        #correct += predicted.eq(target.data).sum()

    max_result.append(correct)

    utils.print_log('# TEST : Epoch : {} | Loss: ({:.4f}) | Acc: ({:.2f}%) ({}/{}) | Err: ({:.2f}%) | Max: ({})'
      .format(epoch, test_loss/(batch_idx+1), 100.*correct/total, correct, total, 100-100.*correct/total, max(max_result)))
    print('# TEST : Epoch : {} | Loss: ({:.4f}) | Acc: ({:.2f}%) ({}/{}) | Err: ({:.2f}% | Max: ({}))'
      .format(epoch, test_loss/(batch_idx+1), 100.*correct/total, correct, total, 100-100.*correct/total, max(max_result)))


def setting_data(data, target, onehot_target):
    data = data.type(torch.cuda.FloatTensor)
    data = utils.renormalize_image(data)
    data = utils.normalize_function(data)
    target = target.type(torch.cuda.FloatTensor)
    target = utils.renormalize_image(target)
    target = utils.normalize_function(target)
    onehot_target = onehot_target(torch.cuda.FloatTensor)
    return data, target, onehot_target


def do_learning(model_dir, number):
    global max_result
    max_result = []
    main(model_dir, number)

if __name__ == '__main__':
    for i in range(20):
        print(str(i)+'for train')	
        model_dir = '../Deep_model/model/{}'.format(i)
        do_learning(model_dir, i)
        