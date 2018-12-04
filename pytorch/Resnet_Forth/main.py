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
import main_model
from collections import OrderedDict


os.environ["CUDA_VISIBLE_DEVICES"] = '6'

def main(main_model_dir, korean_model_dir, number):
    utils.default_model_dir = os.path.join(main_model_dir, 'model')
    BATCH_SIZE = 128
    lr = 0.0002
    EPOCH = 200 
    start_epoch = 0
    start_time = time.time()

    train_Data, test_Data = utils.font_data_onehot_Slice_Loder()
    
    train_loader = torch.utils.data.DataLoader(dataset=train_Data, batch_size=BATCH_SIZE, shuffle=True, num_workers = 4)
    test_loader = torch.utils.data.DataLoader(dataset=test_Data, batch_size=BATCH_SIZE, shuffle=False, num_workers = 4)

    criterion_Cross = nn.CrossEntropyLoss().cuda()
    
    label_model = ResNet()
    korean_checkpoint = utils.load_checkpoint(korean_model_dir)
    new_state_dict = OrderedDict()
    for k, v in korean_checkpoint['state_dict'].items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    label_model.load_state_dict(new_state_dict)

    test(label_model, criterion_Cross, test_loader, 200)
    utils.init_learning(label_model)

    print('conv1.weight.requires_grad', label_model.conv1.weight.requires_grad)

    model = main_model.ResNet(pretrained=label_model)

    print('pretrained.conv1.weight.requires_grad', model.pretrained.conv1.weight.requires_grad)





    if torch.cuda.is_available():
        # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        print("USE", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model).cuda()
        cudnn.benchmark = True

    else:
        print("NO GPU -_-;")
        

    print('MODEL_', model)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion_Cross = nn.CrossEntropyLoss().cuda()

    if not korean_checkpoint:
        pass
    else:
        pass
    # for epoch in range(start_epoch, EPOCH+1):
    #     if epoch < 100:
    #         learning_rate = lr
    #     elif epoch < 150:
    #         learning_rate = lr * 0.1
    #     else:
    #         learning_rate = lr * 0.01
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = learning_rate
    
    #     train(model, optimizer, criterion_Cross, train_loader, epoch)
    #     test(model, criterion_Cross, test_loader, epoch)
    #     utils.save_model_checkpoint(epoch, model, main_model_dir, optimizer)

        
    # utils.conv_weight_L1_printing(model.module)
    now = time.gmtime(time.time() - start_time)
    print('{} hours {} mins {} secs for training'.format(now.tm_hour, now.tm_min, now.tm_sec))


# Train the model
def train(model, optimizer, criterion_Cross , train_loader, epoch):
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
        loss = criterion_Cross(output, target)
        loss.backward(retain_graph=True)
        
        optimizer.step()
        
        print_loss += loss.item()
        _, predicted = torch.max(output, 1)
        total += target.size(0)
        correct += predicted.eq(target.data).sum()
        if batch_idx % 10 == 0:
            utils.print_log('Epoch: {} | Batch: {} |  Cross_Loss: ({:.4f}) | Acc: ({:.2f}%) ({}/{})'
                  .format(epoch, batch_idx, print_loss / (batch_idx + 1) , 100. * correct / total, correct, total))
            print('Epoch: {} | Batch: {} |  Cross_Loss: ({:.4f}) | Acc: ({:.2f}%) ({}/{})'
                  .format(epoch, batch_idx, print_loss / (batch_idx + 1) , 100. * correct / total, correct, total))
            
        
def test(model, criterion_Cross , test_loader, epoch):
    model.eval()
    print_loss = 0
    print_loss2 = 0  
    total = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        if torch.cuda.is_available():
            data, target = Variable(data.cuda()), Variable(target.cuda())
        else:
            data, target = Variable(data), Variable(target)
            
        data, target = setting_data(data, target)
        output = model(data)
        loss = criterion_Cross(output, target )

        print_loss += loss.item()
        _, predicted = torch.max(output, 1)
        total += target.size(0)
        correct += predicted.eq(target.data).sum()
        if batch_idx % 10 == 0:
            utils.print_log('Epoch: {} | Batch: {} |  Cross_Loss: ({:.4f}) | Acc: ({:.2f}%) ({}/{})'
                            .format(epoch, batch_idx, print_loss / (batch_idx + 1), 100. * correct / total, correct, total))
            print('Epoch: {} | Batch: {} |  Cross_Loss: ({:.4f}) | Acc: ({:.2f}%) ({}/{})'
                  .format(epoch, batch_idx, print_loss / (batch_idx + 1), 100. * correct / total, correct, total))


def setting_data(data, target):
    data = data.type(torch.cuda.FloatTensor)
    target = target.type(torch.cuda.LongTensor)
    target = torch.squeeze(target)
    return data, target


def do_learning(main_model_dir, korean_model_dir, number):
    main(main_model_dir, korean_model_dir, number)

if __name__ == '__main__':

    dataset_num = 0

    print('Dataset numper is {}'.format(dataset_num))

    korean_model_dir = '/data2/hhjung/Sonmat_Result/Label_Learning/model'
    main_model_dir = '/data2/hhjung/Sonmat_Result/Resnet_Forth' 

    do_learning(main_model_dir, korean_model_dir, dataset_num)
        