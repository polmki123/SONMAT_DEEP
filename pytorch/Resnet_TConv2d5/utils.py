import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.autograd import Variable
from PIL import Image, ImageFilter
import os
import numpy as np
import glob
import pickle
import gzip
import random
import math
import PIL.ImageOps
default_model_dir = "./"


def make_one_hot() :
    a = np.array([a for a in range(2350)])
    return a

    
def get_num_gen(gen):
    return sum(1 for x in gen)

def is_leaf(model):
    return get_num_gen(model.children()) == 0

def init_learning(model):
    for child in model.children():
        if is_leaf(child):
            if hasattr(child, 'weight'):
                child.weight.requires_grad = False
                # print('True', child)
        else:
            init_learning(child)

def save_model_checkpoint(epoch, model, model_dir, optimizer):
    if epoch % 1 == 0:
        model_filename = 'checkpoint_%02d.pth.tar' % epoch
        save_checkpoint({
            'epoch': epoch,
            'model': model,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, model_filename, model_dir )


def input_Deepmodel_image(inputimagedir):
    frame_dir = '/data2/hhjung/Conpress_Son/unicode_frame/'
    frame_paths = os.listdir(frame_dir)
    input_data = list()
    for frame in frame_paths:
        frame_image = np.array(Image.open( frame_dir + frame)).reshape(1, 64, 64)
        input_image = np.array(Image.open(inputimagedir))
        input_image = np.array(np.split(input_image, 8, axis=1))  # 8*64*64
        Concat_data = np.append(input_image, frame_image, axis=0)
        if ((9, 64, 64) == Concat_data.shape):
            input_data.append(Concat_data)
    
    return input_data, frame_paths

def check_model_result_image(epoch, model, number, model_dir):
    if epoch % 1 == 0:
        saveimagedir = model_dir + '/result_image/' + str(number) + '/' + str(epoch) + '/'
        inputimagedir = '/data2/hhjung/Conpress_Son/test1.jpg'
        input_data, frame_name = input_Deepmodel_image(inputimagedir)
        model.eval()
        
        for number in range(len(input_data)):
            i = np.array(input_data[number])
            i = i.reshape(1, 9, 64, 64)
            input = torch.from_numpy(i)
            input = Variable(input.cuda())
            input = input.type(torch.cuda.FloatTensor)
            input = normalize_image(input)
            output = model(input)
            output = Variable(output[1]).data.cpu().numpy()
            output = output.reshape(64, 64)
            # print(output)
            output =renormalize_image(output)
            img = Image.fromarray(output.astype('uint8'), 'L')
            img = img.filter(ImageFilter.SHARPEN)
            if not os.path.exists(saveimagedir):
                os.makedirs(saveimagedir)
            img.save(saveimagedir + frame_name[number], "PNG")

def check_model_result_image_v2(epoch, model, number, model_dir):
    if epoch % 1 == 0:
        saveimagedir = model_dir + '/result_image/' + str(number) + '/' + str(epoch) + '/'
        inputimagedir = '/data2/hhjung/Conpress_Son/test1.jpg'
        input_data, frame_name = input_Deepmodel_image(inputimagedir)
        model.eval()
        input_data = np.array(input_data)
        train_data = torch.utils.data.TensorDataset(torch.from_numpy(input_data))
        train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=64, shuffle=False, num_workers = 4)
        result_data = []

        for _, (data_set) in enumerate(train_loader) :
        	data_set = np.array(data_set)
            data_set = Variable(data_set.cuda())
            data_set = data_set.type(torch.cuda.FloatTensor)
            data_set = normalize_image(data_set)
            output = model(data_set)
            output = Variable(output[1]).data.cpu().numpy()
            output =renormalize_image(output)
            result.extend(output)
            # print(output)
        
        for i in range(len(result_data)) :
            output = result_data[i]
            img = Image.fromarray(output.astype('uint8'), 'L')
            img = img.filter(ImageFilter.SHARPEN)
            if not os.path.exists(saveimagedir):
                os.makedirs(saveimagedir)
            img.save(saveimagedir + frame_name[i], "PNG")
            
def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def normalize_image(img):
    """
    Make image zero centered and in between (-1, 1)
    """
    normalized = (img / 127.5) - 1.
    return normalized

def normalize_function(img):
    img = (img - img.min()) / (img.max() - img.min())
    img = (img - img.mean()) / (img.std())
    return img

def renormalize_image(img):
    renormalized = (img + 1) * 127.5
    return renormalized

def save_checkpoint(state, filename, model_dir):
    
    model_filename = model_dir + filename
    print(model_filename)
    latest_filename = os.path.join(model_dir, 'latest.txt')

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    with open(latest_filename, 'w') as fout:
        fout.write(model_filename)

    torch.save(state, model_filename)
    print("=> saving checkpoint '{}'".format(model_filename))

    return

def load_checkpoint(model_dir):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    latest_filename = os.path.join(model_dir, 'latest.txt')
    if os.path.exists(latest_filename):
        with open(latest_filename, 'r') as fin:
            model_filename = fin.readlines()[0]
    else:
        return None
    print("=> loading checkpoint '{}'".format(model_filename))
    state = torch.load(model_filename)
    return state

def print_log(text, filename="log.txt"):
    if not os.path.exists(default_model_dir):
        os.makedirs(default_model_dir)
    model_filename = os.path.join(default_model_dir, filename)
    with open(model_filename, "a") as myfile:
        myfile.write(text + "\n")


def font_data_onehot_Slice_Loder():
    # read train data
    data_dir = '/data2/hhjung/Conpress_Son/Font_Conpress/'
    numpy_x = list()
    numpy_label = list()
    load_check = 0
    with gzip.open(data_dir + 'font_train_label.pkl', "rb") as of:
        while True:
            try:
                load_check = load_check + 1
                e = pickle.load(of)
                numpy_x.extend(e[0])
                numpy_label.extend(e[1])
                if len(numpy_x) % 1000 == 0:
                    print("processed %d examples" % len(numpy_x))
            except EOFError:
                print('error')
                break
            except Exception:
                print('error')
                pass
        print("unpickled total %d examples" % len(numpy_x))
    
    X_datas = np.array(numpy_x)
    print(X_datas.shape)
    label_datas = np.array(numpy_label)
    print(label_datas.shape)
    
    # read test data
    numpy_test = list()
    numpy_label_test = list()
    with gzip.open(data_dir + 'font_test_label.pkl', "rb") as of:
        while True:
            try:
                e = pickle.load(of)
                numpy_test.extend(e[0])
                numpy_label_test.extend(e[1])
                if len(numpy_test) % 1000 == 0:
                    print("processed %d examples" % len(numpy_test))
            except EOFError:
                print('error')
                break
            except Exception:
                print('error')
                pass
        print("unpickled total %d examples" % len(numpy_test))
    
    X_test_datas = np.array(numpy_test)
    print(X_test_datas.shape)
    test_label_datas = np.array(numpy_label_test)
    print(test_label_datas.shape)
    # make train, test dataset
    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_datas), torch.from_numpy(label_datas))
    test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_test_datas), torch.from_numpy(test_label_datas))
    
    return train_dataset, test_dataset

def Package_Data_Slice_Loder(number):
    data_dir = '/data2/hhjung/Conpress_Son/'
    numpy_x = list()
    numpy_label = list()
    with gzip.open(data_dir + 'train_' + str(number) +'.pkl', "rb") as of:
        while True:
            try:
                e = pickle.load(of)
                numpy_x.extend(e[0])
                numpy_label.extend(e[1])
                 
                if len(numpy_x) % 1000 == 0:
                    print("processed %d examples" % len(numpy_x))
            except EOFError:
                print('error')
                break
            except Exception:
                print('error')
                pass
        print("unpickled total %d examples" % len(numpy_x))

    X_datas = np.array(numpy_x)
    print(X_datas.shape)
    label_datas = np.array(numpy_label)
    print(label_datas.shape)
    numpy_test = list()
    numpy_label_test = list()
    with gzip.open(data_dir + 'test_' + str(number) + '.pkl', "rb") as of:
        while True:
            try:
                e = pickle.load(of)
                numpy_test.extend(e[0])
                numpy_label_test.extend(e[1])
                if len(numpy_test) % 1000 == 0:
                    print("processed %d examples" % len(numpy_test))
            except EOFError:
                print('error')
                break
            except Exception:
                print('error')
                pass
        print("unpickled total %d examples" % len(numpy_test))
        
    X_test_datas = np.array(numpy_test)
    print(X_test_datas.shape)
    test_label_datas = np.array(numpy_label_test)
    print(test_label_datas.shape)
    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_datas), torch.from_numpy(label_datas))

    test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_test_datas),torch.from_numpy(test_label_datas))
    return train_dataset, test_dataset

def Package_Data_onehot_Slice_Loder(number):
    data_dir = '/data2/hhjung/Conpress_Son/'
    # read train data
    numpy_x = list()
    numpy_label = list()
    numpy_onehot = list()
    load_check = 0
    with gzip.open( data_dir + 'train_' + str(number) + '.pkl', "rb") as of:
        while True:
            try:
                load_check = load_check + 1
                e = pickle.load(of)
                numpy_x.extend(e[0])
                numpy_label.extend(e[1])
                numpy_onehot.extend(make_one_hot())
                if len(numpy_x) % 1000 == 0:
                    print("processed %d examples" % len(numpy_x))
            except EOFError:
                print('error')
                break
            except Exception:
                print('error')
                pass
        print("unpickled total %d examples" % len(numpy_x))
    
    X_datas = np.array(numpy_x)
    print(X_datas.shape)
    label_datas = np.array(numpy_label)
    print(label_datas.shape)
    onehot_datas = np.array(numpy_onehot)
    print(onehot_datas.shape)

    # read test data
    numpy_test = list()
    numpy_label_test = list()
    numpy_onehot_test = list()
    load_check = 0
    with gzip.open(data_dir + 'test_' + str(number) + '.pkl', "rb") as of:
        while True :
            try:
                load_check = load_check + 1
                e = pickle.load(of)
                numpy_test.extend(e[0])
                numpy_label_test.extend(e[1])
                numpy_onehot_test.extend(make_one_hot())
                if len(numpy_test) % 1000 == 0:
                    print("processed %d examples" % len(numpy_test))
            except EOFError:
                print('error')
                break
            except Exception:
                print('error')
                pass
        print("unpickled total %d examples" % len(numpy_test))
    
    X_test_datas = np.array(numpy_test)
    print(X_test_datas.shape)
    test_label_datas = np.array(numpy_label_test)
    print(test_label_datas.shape)
    onehot_test_datas = np.array(numpy_onehot_test)
    print(onehot_test_datas.shape)
    #make train, test dataset
    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_datas), torch.from_numpy(label_datas), torch.from_numpy(onehot_datas))
    test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_test_datas), torch.from_numpy(test_label_datas),  torch.from_numpy(onehot_test_datas))
    
    return train_dataset, test_dataset

def Test_Data_onehot_Slice_Loder(number):
    # read train data
    numpy_x = np.random.rand(2350,9,64,64)
    numpy_label = np.random.rand(2350,1,64,64)
    numpy_onehot = make_one_hot()
    
    X_datas = np.array(numpy_x)
    print(X_datas.shape)
    label_datas = np.array(numpy_label)
    print(label_datas.shape)
    onehot_datas = np.array(numpy_onehot)
    print(onehot_datas.shape)

    # read test data
    numpy_test = np.random.rand(2350,9,64,64)
    numpy_label_test = np.random.rand(2350,1,64,64)
    numpy_onehot_test = make_one_hot()
    
    X_test_datas = np.array(numpy_test)
    print(X_test_datas.shape)
    test_label_datas = np.array(numpy_label_test)
    print(test_label_datas.shape)
    onehot_test_datas = np.array(numpy_onehot_test)
    print(onehot_test_datas.shape)
    #make train, test dataset
    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_datas), torch.from_numpy(label_datas), torch.from_numpy(onehot_datas))
    test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_test_datas), torch.from_numpy(test_label_datas),  torch.from_numpy(onehot_test_datas))
    
    return train_dataset, test_dataset

if __name__ == '__main__':
    Package_Data_onehot_Slice_Loder(1)
