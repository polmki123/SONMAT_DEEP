import torch
import torch.nn as nn
from torchvision import datasets, transforms
from PIL import Image
import os
import numpy as np
import glob
import pickle
import gzip
import random
default_model_dir = "./"

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def normalize_image(img):
    """
    Make image zero centered and in between (-1, 1)
    """
    normalized = (img / 127.5) - 1.
    return normalized

def save_checkpoint(state, filename, model_dir):

    # model_dir = 'drive/app/torch/save_Routing_Gate_2'
    model_filename = os.path.join(model_dir, filename)
    latest_filename = os.path.join(model_dir, 'latest.txt')

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    with open(latest_filename, 'w') as fout:
        fout.write(model_filename)

    torch.save(state, model_filename)
    print("=> saving checkpoint '{}'".format(model_filename))

    return

def load_checkpoint(model_dir):

    # model_dir = 'drive/app/torch/save_Routing_Gate_2'
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


def Package_Data_Slice_Loder():
    numpy_x = list()
    numpy_label = list()
    with gzip.open('../Conpress/train' + str(number) +'.pkl', "rb") as of:
        while True:
            try:
                e = pickle.load(of)
                numpy_x.extend(e[0])
                numpy_label.extend(e[1])
                if len(numpy_x) % 1000 == 0:
                    print("processed %d examples" % len(numpy_x))
            except EOFError:
                break
            except Exception:
                pass
        print("unpickled total %d examples" % len(numpy_x))

    X_datas = np.array(numpy_x)
    label_datas = np.array(numpy_label)

    numpy_test = list()
    numpy_label_test = list()
    with gzip.open('../Conpress/test' + str(number) + '.pkl', "rb") as of:
        while True:
            try:
                e = pickle.load(of)
                numpy_test.extend(e[0])
                numpy_label_test.extend(e[1])
                if len(numpy_test) % 1000 == 0:
                    print("processed %d examples" % len(numpy_test))
            except EOFError:
                break
            except Exception:
                pass
        print("unpickled total %d examples" % len(numpy_test))
        
    X_test_datas = np.array(numpy_test)
    test_label_datas = np.array(numpy_label_test)
    
    
    train_dataset = torch.from_numpy(X_datas), torch.from_numpy(label_datas)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=512,
                                               shuffle=True,
                                               num_workers=4)

    test_dataset = torch.from_numpy(X_test_datas), torch.from_numpy(test_label_datas)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                               batch_size=512,
                                               shuffle=True,
                                               num_workers=4)
    return train_loader, test_loader

def Package_Data_Loder():
    numpy_x = list()
    numpy_label = list()
    with gzip.open('../Conpress/train.pkl', "rb") as of:
        while True:
            try:
                e = pickle.load(of)
                numpy_x.extend(e[0])
                numpy_label.extend(e[1])
                if len(numpy_x) % 1000 == 0:
                    print("processed %d examples" % len(numpy_x))
            except EOFError:
                break
            except Exception:
                pass
        print("unpickled total %d examples" % len(numpy_x))
    
    X_datas = np.array(numpy_x)
    label_datas = np.array(numpy_label)
    
    numpy_test = list()
    numpy_label_test = list()
    with gzip.open('../Conpress/test.pkl', "rb") as of:
        while True:
            try:
                e = pickle.load(of)
                numpy_test.extend(e[0])
                numpy_label_test.extend(e[1])
                if len(numpy_test) % 1000 == 0:
                    print("processed %d examples" % len(numpy_test))
            except EOFError:
                break
            except Exception:
                pass
        print("unpickled total %d examples" % len(numpy_test))
    
    X_test_datas = np.array(numpy_test)
    test_label_datas = np.array(numpy_label_test)
    
    train_dataset = torch.from_numpy(X_datas), torch.from_numpy(label_datas)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=512,
                                               shuffle=True,
                                               num_workers=4)
    
    test_dataset = torch.from_numpy(X_test_datas), torch.from_numpy(test_label_datas)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=512,
                                              shuffle=True,
                                              num_workers=4)
    return train_loader, test_loader

def Data_loader():
    batch_size = 512
    print("data read")
    
    train_dir = '../Deep_model/Train_data/'
    frame_dir = '../Deep_model/frame_label/'
    label_dir = '../Deep_model/label/'

    file_list = os.listdir(train_dir)
    X_datas = []
    label_datas = []
    for file in file_list:
        print(file)
        train_paths = glob.glob(os.path.join(train_dir + file, '*.jpg'))
        label_define = os.listdir(train_dir + file)
        Concat_path = []
        for i in range(len(train_paths)):
            Concat_path.append([train_paths[i], label_define[i]])
        for font_image, label in Concat_path:
            train_image = np.array(Image.open(font_image))  # read 64*512
            train_image = normalize_image(train_image)
            train_datas = np.array(np.split(train_image, 8, axis=1))  # 8*64*64
            frame_paths = glob.glob(os.path.join(frame_dir, "*.jpg"))
            for p in frame_paths:
                frame_datas = np.array(Image.open(p))  # 64*64
                frame_datas = normalize_image(frame_datas)
                frame_datas = frame_datas.reshape(1, 64, 64)  # 1*64*64
                Concat_data = np.append(train_datas, frame_datas, axis=1)  # 9*64*64
                if( (9,64,64) == Concat_data.shape):
                    X_datas.append(Concat_data)
            # X_datas = np.array(X_datas)  # 2350*9*64*64
        
            label_paths = glob.glob(os.path.join(label_dir + label[:-4], "*.jpg"))  # label 데이터 읽기
            
            for p in label_paths:
                label_image = np.array(Image.open(p))  # 64*64
                label_image = normalize_image(label_image)
                label_image = label_image.reshape(1,64, 64) #1*64*64
                if ((1, 64, 64) == label_image.shape):
                    label_datas.append(label_image)
                    
    # label_datas = np.array(label_datas)  # 2350*1*64*64
    X_datas = np.array(X_datas)  # (n*2350)*9*64*64
    label_datas = np.array(label_datas)  # (n*2350)*1*64*64
    
    #x_datas (n*2350)*9*64*64 y_datas (n*2350)*1*64*64
    train_dataset = torch.from_numpy(X_datas) , torch.from_numpy(label_datas)
    # font_Image
    # label_data 가 2350*64*64 이 되게 만들기
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=4)

    return train_loader


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def Data_Slice_loader():
    batch_size = 512
    print("data read")
    
    train_dir = '../Deep_model/Train_data/'
    frame_dir = '../Deep_model/frame_label/'
    label_dir = '../Deep_model/label/'
    
    file_list = os.listdir(train_dir)
    X_datas = []
    label_datas = []
    train_datalist = []
    label_datalist = []
    test_datalist = []
    test_label_datalist = []
    for file in file_list:
        train_paths = glob.glob(os.path.join(train_dir + file, '*.jpg'))
        train_datalist.extend(train_paths[20:])
        test_datalist.extend(train_paths[:20])
        label_define = os.listdir(train_dir + file)
        label_datalist.extend(label_define[20:])
        test_label_datalist.extend(label_define[20:])
    
    Concat_data_tolist = Slice_datalist(label_datalist, train_datalist)
    Concat_test_tolist = Slice_test_datalist(test_datalist, test_label_datalist)
    
    for A in range(len(Concat_data_tolist)):
        train_datapath = list()
        for B in range(len(Concat_data_tolist[A])):
            train_datapath.append(Concat_data_tolist[A][B][0])
        label_datapath = list()
        for B in range(len(Concat_data_tolist[A])):
            label_datapath.append(Concat_data_tolist[A][B][1])
        X_datas = []
        label_datas = []
        for i in range(len(train_datapath)):
            train_image = np.array(Image.open(train_datapath[i]))  # read 64*512
            train_image = normalize_image(train_image)
            train_datas = np.array(np.split(train_image, 8, axis=1))  # 8*64*64
            frame_paths = glob.glob(os.path.join(frame_dir, "*.jpg"))
            for p in frame_paths:
                frame_datas = np.array(Image.open(p))  # 64*64
                frame_datas = normalize_image(frame_datas)
                frame_datas = frame_datas.reshape(1, 64, 64)  # 1*64*64
                Concat_data = np.append(train_datas, frame_datas, axis=1)  # 9*64*64
                if ((9, 64, 64) == Concat_data.shape):
                    X_datas.append(Concat_data)
            # X_datas = np.array(X_datas)  # 2350*9*64*64
            
            label_paths = glob.glob(os.path.join(label_dir + label_datapath[i][:-4], "*.jpg"))  # label 데이터 읽기
            for p in label_paths:
                label_image = np.array(Image.open(p))  # 64*64
                label_image = normalize_image(label_image)
                label_image = label_image.reshape(1, 64, 64)  # 1*64*64
                if ((1, 64, 64) == label_image.shape):
                    label_datas.append(label_image)
                    # label_datas = np.array(label_datas)  # 2350*1*64*64
                    
        X_datas = np.array(X_datas)  # (n*2350)*9*64*64
        label_datas = np.array(label_datas)  # (n*2350)*1*64*64

    for A in range(len(Concat_test_tolist)):
        test_datapath = list()
        for B in range(len(Concat_test_tolist[A])):
            test_datapath.append(Concat_test_tolist[A][B][0])
        test_label_datapath = list()
        for B in range(len(Concat_test_tolist[A])):
            test_label_datapath.append(Concat_test_tolist[A][B][1])
        X_test_datas = []
        test_label_datas = []
        for i in range(len(test_datapath)):
            test_image = np.array(Image.open(test_datapath[i]))  # read 64*512
            test_image = normalize_image(test_image)
            test_datas = np.array(np.split(test_image, 8, axis=1))  # 8*64*64
            frame_paths = glob.glob(os.path.join(frame_dir, "*.jpg"))
            for p in frame_paths:
                frame_datas = np.array(Image.open(p))  # 64*64
                frame_datas = normalize_image(frame_datas)
                frame_datas = frame_datas.reshape(1, 64, 64)  # 1*64*64
                Concat_data = np.append(test_datas, frame_datas, axis=1)  # 9*64*64
                if ((9, 64, 64) == Concat_data.shape):
                    X_test_datas.append(Concat_data)
            #  X_test_datas = np.array( X_test_datas)  # 2350*9*64*64
    
            test_label_paths = glob.glob(os.path.join(label_dir + test_label_datapath[i][:-4], "*.jpg"))  # label 데이터 읽기
            for p in test_label_paths:
                label_image = np.array(Image.open(p))  # 64*64
                label_image = normalize_image(label_image)
                label_image = label_image.reshape(1, 64, 64)  # 1*64*64
                if ((1, 64, 64) == label_image.shape):
                    test_label_datas.append(label_image)
                    # label_datas = np.array(label_datas)  # 2350*1*64*64

        X_test_datas = np.array(X_test_datas)  # (n*2350)*9*64*64
        test_label_datas = np.array(test_label_datas)  # (n*2350)*1*64*64
        
        # x_datas (n*2350)*9*64*64 y_datas (n*2350)*1*64*64
        test_dataset = torch.from_numpy(X_test_datas), torch.from_numpy(test_label_datas)
        train_dataset = torch.from_numpy(X_datas), torch.from_numpy(label_datas)
        # font_Image
        # label_data 가 2350*64*64 이 되게 만들기
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=4)

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=4)

    return train_loader, test_loader


def Slice_test_datalist(test_datalist, test_label_datalist):
    Test_Concat_datalist = []
    for i in range(len(test_datalist)):
        Test_Concat_datalist.append([test_datalist[i], test_label_datalist[i]])
    Test_Concat_data_tolist = []
    for group in chunker(Test_Concat_datalist, 8):
        Test_Concat_data_tolist.append(group)
    
    return Test_Concat_data_tolist


def Slice_datalist(label_datalist, train_datalist):
    Concat_datalist = []
    for i in range(len(train_datalist) - 8):
        Concat_datalist.append([train_datalist[i], label_datalist[i]])
    random.shuffle(Concat_datalist)
    Concat_data_tolist = []
    for group in chunker(Concat_datalist, 42):
        Concat_data_tolist.append(group)
    return Concat_data_tolist


if __name__ == '__main__':
    A = Data_loader()
    print('end')
    print(A.shape)
    