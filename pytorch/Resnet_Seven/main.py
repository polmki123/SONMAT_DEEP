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
    BATCH_SIZE = 64
    lr = 0.001
    EPOCH = 15
    start_epoch = 0
    train_Data, test_Data = utils.Package_Data_onehot_Slice_Loder(number)
    
    train_loader = torch.utils.data.DataLoader(dataset=train_Data, batch_size=BATCH_SIZE, shuffle=True, num_workers = 4)
    test_loader = torch.utils.data.DataLoader(dataset=test_Data, batch_size=BATCH_SIZE, shuffle=False, num_workers = 4)

    
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
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas = (0.5, 0.999))
    criterion_MSE = nn.MSELoss().cuda()
    criterian_first = nn.CrossEntropyLoss().cuda()
    criterian_middle = nn.CrossEntropyLoss().cuda()
    checkpoint = utils.load_checkpoint(utils.default_model_dir)

    if not checkpoint:
        pass
    else:
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    for epoch in range(start_epoch, EPOCH+1):
        if epoch < 6:
            learning_rate = lr
        elif epoch < 10:
            learning_rate = lr * 0.1
        else:
            learning_rate = lr * 0.01
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate
    
        train(model, optimizer, criterion_MSE, criterian_first, criterian_middle, train_loader, epoch)
        test(model, criterion_MSE, criterian_first ,criterian_middle, test_loader, epoch)
        utils.save_model_checkpoint(epoch, model, utils.default_model_dir, optimizer)
        utils.check_model_result_image(epoch, model, number, model_dir)
        
    # utils.conv_weight_L1_printing(model.module)
    now = time.gmtime(time.time() - start_time)
    print('{} hours {} mins {} secs for training'.format(now.tm_hour, now.tm_min, now.tm_sec))


# Train the model
def train(model, optimizer, criterion_MSE, criterian_first, criterian_middle, train_loader, epoch):
    model.train()
    print_loss = 0
    print_loss2 = 0  
    print_loss3 = 0
    total = 0
    correct = 0
    correct2 = 0
    for batch_idx, (data, target, onehot_target) in enumerate(train_loader):
        if torch.cuda.is_available():
            data, target, onehot_target = Variable(data.cuda()), Variable(target.cuda()), Variable(onehot_target.cuda())
        else:
            data, target, onehot_target  = Variable(data), Variable(target), Variable(onehot_target)
            
        data, target, onehot_target = setting_data(data, target, onehot_target)
        optimizer.zero_grad()
        output = model(data)

        first_cross = criterian_first(output[0], onehot_target)
        Second_cross = criterian_middle(output[2], onehot_target)
        image_loss = criterion_MSE(output[1], target)

        first_cross.backward(retain_graph=True)
        Second_cross.backward(retain_graph=True)
        image_loss.backward(retain_graph=True)
        optimizer.step()

        print_loss += first_cross.item()
        print_loss2 += image_loss.item()
        print_loss3 += Second_cross.item()
        _, predicted = torch.max(output[0].data, 1)
        _, predicted2 = torch.max(output[2].data, 1)
        total += target.size(0)
        correct += predicted.eq(onehot_target.data).sum()
        correct2 += predicted2.eq(onehot_target.data).sum()
        if batch_idx % 10 == 0:
            utils.print_log('Epoch: {} | Batch: {} |  Cross_Loss: ({:.4f}) | image_Loss: ({:.4f}) | Acc: ({:.2f}%) ({}/{}) | Acc2 : ({:.2f}%) ({}/{})'
                  .format(epoch, batch_idx, print_loss / (batch_idx + 1), print_loss2 , 100. * correct / total, correct, total, 100. * correct2 / total, correct2, total))
            print('Epoch: {} | Batch: {} |  Cross_Loss: ({:.4f}) | image_Loss: ({:.4f}) | Acc: ({:.2f}%) ({}/{}) | Acc2 : ({:.2f}%) ({}/{})'
                  .format(epoch, batch_idx, print_loss / (batch_idx + 1), print_loss2 , 100. * correct / total, correct, total, 100. * correct2 / total, correct2, total))
            
        
def test(model, criterion_MSE, criterian_first ,criterian_middle, test_loader, epoch):
    model.eval()
    print_loss = 0
    print_loss2 = 0  
    print_loss3 = 0
    total = 0
    correct = 0
    correct2 = 0
    for batch_idx, (data, target, onehot_target) in enumerate(test_loader):
        if torch.cuda.is_available():
            data, target, onehot_target = Variable(data.cuda()), Variable(target.cuda()), Variable(onehot_target.cuda())
        else:
            data, target, onehot_target = Variable(data), Variable(target), Variable(onehot_target)
            
        data, target, onehot_target = setting_data(data, target, onehot_target)
        output = model(data)

        first_cross = criterian_first(output[0], onehot_target)
        Second_cross = criterian_middle(output[2], onehot_target)
        image_loss = criterion_MSE(output[1], target)



        print_loss += first_cross.item()
        print_loss2 += image_loss.item()
        print_loss3 += Second_cross.item()
        _, predicted = torch.max(output[0].data, 1)
        _, predicted2 = torch.max(output[2].data, 1)
        total += target.size(0)
        correct += predicted.eq(onehot_target.data).sum()
        correct2 += predicted2.eq(onehot_target.data).sum()
        if batch_idx % 10 == 0:
            utils.print_log('#TEST Epoch: {} | Batch: {} |  Cross_Loss: ({:.4f}) | image_Loss: ({:.4f}) | Acc: ({:.2f}%) ({}/{}) | Acc2 : ({:.2f}%) ({}/{})'
                  .format(epoch, batch_idx, print_loss / (batch_idx + 1), print_loss2 , 100. * correct / total, correct, total, 100. * correct2 / total, correct2, total))
            print('#TEST poch: {} | Batch: {} |  Cross_Loss: ({:.4f}) | image_Loss: ({:.4f}) | Acc: ({:.2f}%) ({}/{}) | Acc2 : ({:.2f}%) ({}/{})'
                  .format(epoch, batch_idx, print_loss / (batch_idx + 1), print_loss2 , 100. * correct / total, correct, total, 100. * correct2 / total, correct2, total))
            


def setting_data(data, target, onehot_target):
    data = data.type(torch.cuda.FloatTensor)
    target = target.type(torch.cuda.FloatTensor)
    onehot_target = onehot_target.type(torch.cuda.LongTensor)
    onehot_target = torch.squeeze(onehot_target)
    return data, target, onehot_target


def do_learning(model_dir, number):
    global max_result
    max_result = []
    main(model_dir, number)

if __name__ == '__main__':
    print(str(1)+'for train')	
    model_dir = '/data2/hhjung/Sonmat_Result/Resnet_Seven' 
    do_learning(model_dir, 1)
        
