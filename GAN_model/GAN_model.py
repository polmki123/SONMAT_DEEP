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

# Train the model
# def train(model, optimizer, criterion, train_loader, epoch):
#     model.train()
#     train_loss = 0
#     total = 0
#     correct = 0
#     for batch_idx, (data, target) in enumerate(train_loader):
#         #print(batch_idx)
#     #for data, target in train_loader:
#         if torch.cuda.is_available():
#             data, target = Variable(data.cuda()), Variable(target.cuda())
#         else:
#             data, target = Variable(data), Variable(target)
#         data = renormalize_image(data)
#         data = normalize_function(data)
#         target = target.type(torch.cuda.FloatTensor)
#         target = renormalize_image(target)
#         target = normalize_function(target)
#         optimizer.zero_grad()
#         output = model(data)
#         loss = criterion(output, target)
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.item()
#         _, predicted = torch.max(output.data, 1)
#         total += target.size(0)
#         #correct += predicted.eq(target.data).sum()
#         if batch_idx % 10 == 0:
#             utils.print_log('Epoch: {} | Batch: {} |  Loss: ({:.4f}) | Acc: ({:.2f}%) ({}/{})'
#                   .format(epoch, batch_idx, train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
#             print('Epoch: {} | Batch: {} |  Loss: ({:.4f}) | Acc: ({:.2f}%) ({}/{})'
#                   .format(epoch, batch_idx, train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
            
        
# def test(model, criterion, test_loader, epoch):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     total = 0
#     for batch_idx, (data, target) in enumerate(test_loader):
#         if torch.cuda.is_available():
#             data, target = Variable(data.cuda()), Variable(target.cuda())
#         else:
#             data, target = Variable(data), Variable(target)
#         data = data.type(torch.cuda.FloatTensor)
#         data = renormalize_image(data)
#         data = normalize_function(data)
#         target = target.type(torch.cuda.FloatTensor)
#         target = renormalize_image(target)
#         target = normalize_function(target)
#         outputs = model(data)
#         loss = criterion(outputs, target)

#         test_loss += loss.item()
#         _, predicted = torch.max(outputs.data, 1)
#         total += target.size(0)
#         #correct += predicted.eq(target.data).sum()

#     max_result.append(correct)

#     utils.print_log('# TEST : Epoch : {} | Loss: ({:.4f}) | Acc: ({:.2f}%) ({}/{}) | Err: ({:.2f}%) | Max: ({})'
#       .format(epoch, test_loss/(batch_idx+1), 100.*correct/total, correct, total, 100-100.*correct/total, max(max_result)))
#     print('# TEST : Epoch : {} | Loss: ({:.4f}) | Acc: ({:.2f}%) ({}/{}) | Err: ({:.2f}% | Max: ({}))'
#       .format(epoch, test_loss/(batch_idx+1), 100.*correct/total, correct, total, 100-100.*correct/total, max(max_result)))


def main(model_dir, number):
    BATCH_SIZE = 128
    EPOCH = 200 
    lr = 0.0002
    train_Data, test_Data = utils.Package_Data_Slice_Loader(number+1)
    
    train_loader = torch.utils.data.DataLoader(dataset=train_Data, batch_size=BATCH_SIZE, shuffle=True, num_workers = 4)
    test_loader = torch.utils.data.DataLoader(dataset=test_Data, batch_size=BATCH_SIZE, shuffle=False, num_workers = 4)
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
    
    real_label = 1
    fake_label = 0
    # Loss and Optimizer
    Loss= tnn.BCELoss().cuda() if torch.cuda.is_available() else tnn.BCELOSS()
    optimizerD = torch.optim.Adam(Discri.parameters(), lr = 0.0002,betas = (0.5, 0.999))
    optimizerG = torch.optim.Adam(Gener.parameters(), lr = 0.0002, betas = (0.5, 0.999))
    Gener_checkpoint = utils.Gener_load_checkpoint(model_dir+str(number))
    Discri_checkpoint = utils.Discri_load_checkpoint(model_dir+str(number))

    if not Gener_checkpoint:
        pass
    else:
        #start_epoch = checkpoint['epoch'] + 1
        Gener.load_state_dict(checkpoint['state_dict'])
        optimizerG.load_state_dict(checkpoint['optimizer'])

    if not Discri_checkpoint:
        pass
    else:
        #start_epoch = checkpoint['epoch'] + 1
        Discri.load_state_dict(checkpoint['state_dict'])
        optimizerD.load_state_dict(checkpoint['optimizer'])

    for epoch in range(1, EPOCH+1):
        loss_D = 0.0

        if epoch < 100:
            learning_rate = lr
        elif epoch < 150:
            learning_rate = lr * 0.1
        else:
            learning_rate = lr * 0.01

        for param_group in optimizerD.param_groups:
            param_group['lr'] = learning_rate

        for param_group in optimizerG.param_groups:
            param_group['lr'] = learning_rate


        for batch_idx, (data, target) in enumerate(train_loader):
            Gener.train()
            Discri.train()
            data = renormalize_image(data)
            data = normalize_function(data)
            data = data.type(torch.cuda.FloatTensor)
            #real
            target = renormalize_image(target)
            target = normalize_function(target)
            target = target.type(torch.cuda.FloatTensor)
            # Make noise
            b_size = len(data) 
            noise = torch.randn(b_size, 1, 64, 64).uniform_(0,1)
            noise = noise.type(torch.cuda.FloatTensor)
            ####### Train d to recognize color image as real
            output = Discri(target,b_size)
            real_loss = torch.mean((output-1)**2)

            ###### Train d to recognize fake image as fake
            Data_noise = (torch.cat([data,noise],dim=1))
            Data_noise = Data_noise.type(torch.cuda.FloatTensor)
            fake_img = Gener(Data_noise)
            output = Discri(fake_img,b_size)
            fake_loss = torch.mean(output**2)

            ###### Backpro & Optim D
            d_loss = real_loss + fake_loss
            Discri.zero_grad()
            Gener.zero_grad()
            d_loss.backward()
            optimizerD.step()
        
            ######## Train Generator ########
            noise2 = torch.randn(b_size, 1, 64, 64).uniform_(0,1)
            noise2 = noise2.type(torch.cuda.FloatTensor)
            Data_noise2 = (torch.cat([data,noise2],dim=1))
            Data_noise2 = Data_noise2.type(torch.cuda.FloatTensor)
            fake_img = Gener(Data_noise2)
            output = Discri(fake_img,b_size)
            g_loss = torch.mean((output-1)**2)
            
            ###### Backpro & Optim G
            Discri.zero_grad()
            Gener.zero_grad()
            g_loss.backward()
            optimizerG.step()
            if batch_idx % 20 ==0 :
                print('[epoch:%d, batch:%5d] real loss: %.4f, fake_loss : %.4f, g_loss : %.4f' % (epoch ,batch_idx, real_loss.item(),fake_loss.item(), g_loss.item() ))
        if epoch % 20 == 0:
            Gener_filename = '/Gener_checkpoint_%02d.pth.tar' % epoch
            utils.Gener_save_checkpoint({
                'epoch': epoch,
                'model': Gener,
                'state_dict': Gener.state_dict(),
                'optimizer': optimizerG.state_dict(),
            }, Gener_filename, model_dir+str(number+1))

            Discri_filename = '/Discri_checkpoint_%02d.pth.tar' % epoch
            utils.Discri_save_checkpoint({
                'epoch': epoch,
                'model': Discri,
                'state_dict': Discri.state_dict(),
                'optimizer': optimizerD.state_dict(),
            }, Discri_filename, model_dir+str(number+1))

        if epoch % 10 == 0:
            saveimagedir = '../GAN_model/save_font_image/' + str(number) + '/' + str(epoch) + '/'
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
                output = (output)
                img = Image.fromarray(output.astype('uint8'), 'L')
                #img = PIL.ImageOps.invert(img)
                if not os.path.exists(saveimagedir):
                    os.makedirs(saveimagedir)
                img.save(saveimagedir + str(check_point) + 'my.jpg')
       
    # utils.conv_weight_L1_printing(model.module)
    now = time.gmtime(time.time() - start_time)
    #print(fake.shape)
    print('{} hours {} mins {} secs for training'.format(now.tm_hour, now.tm_min, now.tm_sec))


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

def do_learning(model_dir, number):
    global max_result
    max_result = []
    main(model_dir, number)

if __name__ == '__main__':
    for i in range(20):
        print(str(i)+'for train')	
        model_dir = '../GAN_model/model/'
        do_learning(model_dir, i)
        


 




