3# -*- coding: utf-8 -*-

import argparse
import glob
import os
import pickle
import random
import numpy as np
from PIL import Image
import imageio
import scipy.misc as misc
import numpy as np
import io
import gzip
def pad_seq(seq, batch_size):
    # pad the sequence to be the multiples of batch_size
    seq_len = len(seq)
    if seq_len % batch_size == 0:
        return seq
    padded = batch_size - (seq_len % batch_size)
    seq.extend(seq[:padded])
    return seq


def normalize_image(img):
    """
    Make image zero centered and in between (-1, 1)
    """
    normalized = (img / 127.5) - 1.
    return normalized


def save_concat_images(imgs, img_path):
    concated = np.concatenate(imgs, axis=1)
    misc.imsave(img_path, concated)


def compile_frames_to_gif(frame_dir, gif_file):
    frames = sorted(glob.glob(os.path.join(frame_dir, "*.png")))
    print(frames)
    images = [misc.imresize(imageio.imread(f), interp='nearest', size=0.33) for f in frames]
    imageio.mimsave(gif_file, images, duration=0.1)
    return gif_file


def Slice_test_datalist(test_datalist, test_label_datalist):
    Test_Concat_datalist = []
    for i in range(len(test_datalist)):
        Test_Concat_datalist.append([test_datalist[i], test_label_datalist[i]])
    Test_Concat_test_tolist = []
    for group in chunker(Test_Concat_datalist, 8):
        Test_Concat_test_tolist.append(group)
    
    return Test_Concat_test_tolist

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def Slice_datalist(label_datalist, train_datalist):
    Concat_datalist = []
    for i in range(len(train_datalist) - 8):
        Concat_datalist.append([train_datalist[i], label_datalist[i]])
    random.shuffle(Concat_datalist)
    Concat_data_tolist = []
    for group in chunker(Concat_datalist, 42):
        Concat_data_tolist.append(group)
    return Concat_data_tolist

def pickle_slice_examples(package_dir):
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
    number = 0
    for A in range(len(Concat_data_tolist)):
        train_datapath = list()
        for B in range(len(Concat_data_tolist[A])):
            train_datapath.append(Concat_data_tolist[A][B][0])
        label_datapath = list()
        for B in range(len(Concat_data_tolist[A])):
            label_datapath.append(Concat_data_tolist[A][B][1])
        number = number + 1
        with gzip.open(package_dir + 'train_' + str(number) +'.pkl', 'wb') as ft:
            for i in range(len(train_datapath)):
                X_datas = []
                label_datas = []
                train_image = np.array(Image.open(train_datapath[i]))  # read 64*512
                train_image = normalize_image(train_image)
                train_datas = np.array(np.split(train_image, 8, axis=1))  # 8*64*64
                frame_paths = glob.glob(os.path.join(frame_dir, "*.jpg"))
                for p in frame_paths:
                    frame_datas = np.array(Image.open(p))  # 64*64
                    frame_datas = normalize_image(frame_datas)
                    frame_datas = frame_datas.reshape(1, 64, 64)  # 1*64*64
                    Concat_data = np.append(train_datas, frame_datas, axis=0)  # 9*64*64
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
                print(X_datas.shape)
                label_datas = np.array(label_datas)  # (n*2350)*1*64*64
                print(label_datas.shape)
                example = (X_datas, label_datas)
                pickle.dump(example, ft)
            ft.close()

    number = 0
    for A in range(len(Concat_test_tolist)):
        test_datapath = list()
        for B in range(len(Concat_test_tolist[A])):
            test_datapath.append(Concat_test_tolist[A][B][0])
        test_label_datapath = list()
        for B in range(len(Concat_test_tolist[A])):
            test_label_datapath.append(Concat_test_tolist[A][B][1])
        number = number + 1
        with gzip.open(package_dir + 'test_' + str(number) + '.pkl', 'wb') as ft:
            for i in range(len(test_datapath)):
                X_test_datas = []
                test_label_datas = []
                test_image = np.array(Image.open(test_datapath[i]))  # read 64*512
                test_image = normalize_image(test_image)
                test_datas = np.array(np.split(test_image, 8, axis=1))  # 8*64*64
                frame_paths = glob.glob(os.path.join(frame_dir, "*.jpg"))
                for p in frame_paths:
                    frame_datas = np.array(Image.open(p))  # 64*64
                    frame_datas = normalize_image(frame_datas)
                    frame_datas = frame_datas.reshape(1, 64, 64)  # 1*64*64
                    Concat_data = np.append(test_datas, frame_datas, axis=0)  # 9*64*64
                    if ((9, 64, 64) == Concat_data.shape):
                        X_test_datas.append(Concat_data)
                # X_test_datas = np.array( X_test_datas)  # 2350*9*64*64
                test_label_paths = glob.glob(os.path.join(label_dir + test_label_datapath[i][:-4], "*.jpg"))  # label 데이터 읽기
                for p in test_label_paths:
                    label_image = np.array(Image.open(p))  # 64*64
                    label_image = normalize_image(label_image)
                    label_image = label_image.reshape(1, 64, 64)  # 1*64*64
                    if ((1, 64, 64) == label_image.shape):
                        test_label_datas.append(label_image)
                        # label_datas = np.array(label_datas)  # 2350*1*64*64
                X_test_datas = np.array(X_test_datas)  # (n*2350)*9*64*64
                print(X_test_datas)
                test_label_datas = np.array(test_label_datas)  # (n*2350)*1*64*64
                print(test_label_datas)
                example = (X_test_datas, test_label_datas)
                pickle.dump(example, ft)
            ft.close()

        
def pickle_examples(package_dir):
    """
    Compile a list of examples into pickled format, so during
    the training, all io will happen in memory
    """
    train_dir= '../Deep_model/Train_data/'
    frame_dir = '../Deep_model/frame_label/'
    label_dir = '../Deep_model/label/'
    
    file_list = os.listdir(train_dir)
    with gzip.open(package_dir, 'wb') as ft:
        for file in file_list:
            print(file)
            train_paths = glob.glob(os.path.join(train_dir+ file, '*.jpg'))
            label_define = os.listdir(train_dir+ file)
            Concat_path = []
            for i in range(len(train_paths)):
                Concat_path.append([train_paths[i], label_define[i]])
            for font_image, label in Concat_path:
                X_datas = []
                train_image = np.array(Image.open(font_image))  # read 64*512
                train_image = normalize_image(train_image)
                train_datas = np.array(np.split(train_image, 8, axis=1))  # 8*64*64
                frame_paths = glob.glob(os.path.join(frame_dir, "*.jpg"))
                for p in frame_paths:
                    frame_datas = np.array(Image.open(p))  # 64*64
                    frame_datas = normalize_image(frame_datas)
                    frame_datas = frame_datas.reshape(1,64,64) # 1*64*64
                    Concat_data = np.append(train_datas, frame_datas, axis=0) # 9*64*64
                    if ((9, 64, 64) == Concat_data.shape):
                        X_datas.append(Concat_data)
                X_datas = np.array(X_datas) # 2350*9*64*64
                print(X_datas.shape)
                label_paths = glob.glob(os.path.join(label_dir + label[:-4], "*.jpg"))  # label 데이터 읽기
                label_datas = []
                for p in label_paths:
                    label_image = np.array(Image.open(p))  # 64*64
                    label_image = normalize_image(label_image)
                    label_image = label_image.reshape(1,64,64) #64*64*1
                    label_datas.append(label_image)
                label_datas = np.array(label_datas)  # 2350*1*64*64
                print(label_datas.shape)
                # x_datas n*(2350*9*64*64) y_datas n*2350*1*64*64
                examples = (X_datas, label_datas)
                pickle.dump(examples, ft, pickle.HIGHEST_PROTOCOL)
        ft.close()
        

if __name__ == "__main__":
    package_dir = 'E:/Conpress/'
    pickle_slice_examples(package_dir)
    