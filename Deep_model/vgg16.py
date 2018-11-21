import torch
import torch.nn as tnn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
#from . import utils
import utils
import os
import time
import glob
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

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



# Train the model
def train(model, optimizer, criterion, train_loader, epoch):
    model.train()
    train_loss = 0
    total = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        #print(batch_idx)
    #for data, target in train_loader:
        if torch.cuda.is_available():
            data, target = Variable(data.cuda()), Variable(target.cuda())
        else:
            data, target = Variable(data), Variable(target)
        data = data.type(torch.cuda.FloatTensor)
        #print(data.type())
        target = target.type(torch.cuda.FloatTensor)
        optimizer.zero_grad()
        #assert (data >= 0. & data <= 1.).all()
        output = model(data)
        loss = criterion(output, target)
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
            
        
def test(model, criterion, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        if torch.cuda.is_available():
            data, target = Variable(data.cuda()), Variable(target.cuda())
        else:
            data, target = Variable(data), Variable(target)
        data = data.type(torch.cuda.FloatTensor)
        target = target.type(torch.cuda.FloatTensor)
        outputs = model(data)
        loss = criterion(outputs, target)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        #correct += predicted.eq(target.data).sum()

    max_result.append(correct)

    utils.print_log('# TEST : Epoch : {} | Loss: ({:.4f}) | Acc: ({:.2f}%) ({}/{}) | Err: ({:.2f}%) | Max: ({})'
      .format(epoch, test_loss/(batch_idx+1), 100.*correct/total, correct, total, 100-100.*correct/total, max(max_result)))
    print('# TEST : Epoch : {} | Loss: ({:.4f}) | Acc: ({:.2f}%) ({}/{}) | Err: ({:.2f}% | Max: ({}))'
      .format(epoch, test_loss/(batch_idx+1), 100.*correct/total, correct, total, 100-100.*correct/total, max(max_result)))


def main(model_dir, number):
    BATCH_SIZE = 128
    LEARNING_RATE = 0.01
    EPOCH = 200
    
    train_Data, test_Data = utils.Package_Data_Slice_Loder(number+1)
    #print(train_Data.shape)
    # testData = dsets.ImageFolder('../data/imagenet/test', transform)
    
    train_loader = torch.utils.data.DataLoader(dataset=train_Data, batch_size=BATCH_SIZE, shuffle=True, num_workers = 4)
    test_loader = torch.utils.data.DataLoader(dataset=test_Data, batch_size=BATCH_SIZE, shuffle=False, num_workers = 4)
    #dataiter = iter(train_loader)
    #train_test, test_test = dataiter.next()
    #print(train_test.shape)    
    utils.default_model_dir = model_dir
    lr = 0.1
    start_time = time.time()

    model = VGG16()
    #model.cuda()
    
    if torch.cuda.is_available():
        # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        print("USE", torch.cuda.device_count(), "GPUs!")
        model = tnn.DataParallel(model).cuda()
        cudnn.benchmark = True

    else:
        print("NO GPU -_-;")
        
    # Loss and Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    criterion = tnn.BCEWithLogitsLoss().cuda()

    start_epoch = 0
    checkpoint = utils.load_checkpoint(model_dir+str(number-1))

    if not checkpoint:
        pass
    else:
        #start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    for epoch in range(0, EPOCH):
        if epoch < 150:
            learning_rate = lr
        elif epoch < 225:
            learning_rate = lr * 0.1
        else:
            learning_rate = lr * 0.01
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate
    
        train(model, optimizer, criterion, train_loader, epoch)
        test(model, criterion, test_loader, epoch)
        
        if epoch % 20 == 0:
            model_filename =  '/checkpoint_%02d.pth.tar' % epoch
            utils.save_checkpoint({
                'epoch': epoch,
                'model': model,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, model_filename, model_dir+str(number))
        if epoch % 100 == 0:
            saveimagedir = '../Deep_model/save_font_image/' + str(number) + '/' + str(epoch/100) + '/'
            inputimagedir = '../Deep_model/Test1.jpg'
            input_data = input_Deepmodel_image(inputimagedir)
            model.eval()
            number = 0
            for i in input_data:
                number = number + 1
                i = np.array(i)
                i = i.reshape(1,9,64,64)
                input = torch.from_numpy(i)
                input = Variable(input.cuda())
                input = input.type(torch.cuda.FloatTensor)
                output = model(input)
                output = Variable(output).data.cpu().numpy()
                output = output.reshape(64,64,1)
                output = renormalize_image(output)
                img = Image.fromarray(output.astype('unit'), 'L')
                img.save(saveimagedir + str(number) + 'my.jpg')
       
    # utils.conv_weight_L1_printing(model.module)
    now = time.gmtime(time.time() - start_time)
    print('{} hours {} mins {} secs for training'.format(now.tm_hour, now.tm_min, now.tm_sec))

def input_Deepmodel_imge(inputimagedir) :
    frame_dir = '../Deep_model/frame_label/'
    frame_paths = glob.glob(os.path.join(frame_dir, '*.jpg'))
    input_data = list()
    for frame in frame_paths :
        frame_image = np.array(Image.open(frame)).reshape(1,64,64)
        frame_image = normalize_image(frame_image)
        #print(frame_image.shape)
        input_image = np.array(Image.open(inputimagedir))
        input_image = normalize_image(input_image)
        input_image = np.array(np.split(input_image, 8, axis=1))  # 8*64*64
        #print(input_image.shape)
        Concat_data = np.append(input_image, frame_image, axis=0)
        if ((9, 64, 64) == Concat_data.shape):
            input_data.append(Concat_data)
            
    return input_data

def do_learning(model_dir, number):
    global max_result
    max_result = []
    main(model_dir, number)

if __name__ == '__main__':
    for i in range(20):
        print(str(i)+'for train')	
        model_dir = '../Deep_model/model/'
        do_learning(model_dir, i)
        
