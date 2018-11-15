import numpy as np
from PIL import Image
import cv2
import os
import glob
import pickle
import gzip
import random
# h = Image.open('noname01.bmp')
# h = h.convert('L')
# h = np.array(h)
# print(h)
# print(h.shape)
#
# x = [1,2,3,4,5,6]
# y = [1,2,3,4,5,6]
# new = []
# for i in range(len(x)):
#     new.append([x[i],y[i]])
#
# for a,b in new:
#     print(a)
#     print(b)


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
        train_datalist.extend(train_paths[20 :])
        test_datalist.extend(train_paths[:20])
        label_define = os.listdir(train_dir + file)
        label_datalist.extend(label_define[20:])
        test_label_datalist.extend(label_define[20:])

    Concat_data_tolist = Slice_datalist(label_datalist, train_datalist)
    Concat_test_tolist = Slice_test_datalist(test_datalist, test_label_datalist)
    
    for train_datapath, label_datapath in Concat_data_tolist :
        X_datas = []
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

number = 0

path = '../Deep_model/test.pkl'
# x = np.arange(0,32768)
# x = x.reshape([64,512])
# x = np.array(np.split(x, 8, axis=1))
#
# y = np.arange(0,4096)
# y = y.reshape(1,64,64)
# print(y.shape)
# k = np.append(x,y,axis=0)
# k = k.T
# print(k.shape)
# l = []
# l.append(k)
# l.append(k)
# l.append(k)
# l = np.array(l)
# print(l.shape)
# #
# with gzip.open( path, 'wb') as ft:
#     examples = list()
#     for i in range(10) :
#         # a =l.tolist()
#         # b= x.tolist()
#         examples = (l, x)
#         pickle.dump(examples, ft )
#     ft.close()

with gzip.open( path, 'rb') as ft:
    numpy_x = list()
    numpy_label = list()
    while True:
        try:
            e = pickle.load(ft)
            numpy_x.extend(e[0])
            numpy_label.extend(e[1])
        except EOFError:
            break
        except Exception:
            pass
ft.close()
# numpy_x = examples[:, 0]
# numpy_label = examples[:, 1]
numpy_x = np.array(numpy_x)
numpy_label = np.array(numpy_label)
print(numpy_label.shape)
print(numpy_x.shape)

# x = x.reshape([8,64,64])
# print(x)

# print(x)
# a = [1,2,3,4,5,6,7]
# a = np.asarray(h)

    