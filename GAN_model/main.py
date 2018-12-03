import torch
import torch.nn as nn
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
from model  import * 

os.environ["CUDA_VISIBLE_DEVICES"] = '6'

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:         # Conv weight init
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:  # BatchNorm weight init
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def main(model_dir, number):

    cuda = True if torch.cuda.is_available() else False

    BATCH_SIZE = 128
    EPOCH = 200 
    lr = 0.0002
    train_Data, test_Data = utils.Package_Data_Slice_Loader(number+1)
    
    train_loader = torch.utils.data.DataLoader(dataset=train_Data, batch_size=BATCH_SIZE, shuffle=True, num_workers = 4)
    test_loader = torch.utils.data.DataLoader(dataset=test_Data, batch_size=BATCH_SIZE, shuffle=False, num_workers = 4)
    
    generator = Generator()
    discriminator = Discriminator()

    # Loss function
    adversarial_loss = torch.nn.BCELoss()
    
    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()

    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    utils.default_model_dir = model_dir

    start_time = time.time()
    
    real_label = 1
    fake_label = 0
    # Loss and Optimizer
    Loss= nn.BCELoss().cuda() 
    optimizerD = torch.optim.Adam(discriminator.parameters(), lr = 0.0002,betas = (0.5, 0.999))
    optimizerG = torch.optim.Adam(generator.parameters(), lr = 0.0002, betas = (0.5, 0.999))
    
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

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    for epoch in range(0, EPOCH+1):
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

            # Adversarial ground truths
            valid = Variable(Tensor(target.shape[0], 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(target.shape[0], 1).fill_(0.0), requires_grad=False)

            Gener.train()
            Discri.train()

             #real
            data = Variable(data.type(torch.cuda.FloatTensor))
            real_imgs = Variable(target.type(torch.cuda.FloatTensor))
            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()
            gen_imgs = generator(data)

            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)
            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            if batch_idx % 20 ==0 :
                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, opt.n_epochs, i, len(dataloader),
                                                            d_loss.item(), g_loss.item()))
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
            generator.eval()
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
                output = generator(Data_noise)
                output = Variable(output).data.cpu().numpy()
                output = output.reshape(64,64)
                #print(output)
                output = renormalize_image(output)
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
    
    print(str(0)+'for train')	
    model_dir = '../GAN_model/model/'
    do_learning(model_dir, 0)
        


 




