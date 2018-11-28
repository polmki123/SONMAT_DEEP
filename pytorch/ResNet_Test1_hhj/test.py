import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.autograd import Variable
from PIL import Image
import os
import numpy as np
import glob
import pickle
import gzip
import random
import math
import PIL.ImageOps
import utils
default_model_dir = "./"


def Second_Package_Data_onehot_Slice_Loder():
    # read train data
    numpy_x = list()
    numpy_label = list()
    numpy_onehot = list()
    number = 0

    with gzip.open('../ResNet_Test1/Conpress/Resnet_test.pkl', "rb") as of:
        while number < 10:
            number = number + 1
            try:
                e = pickle.load(of)
                print(e[0].shape)
                print(e[1].shape)
                check = e[0][0][0].reshape(64,64)
                check = utils.renormalize_image(check)
                check2 = e[0][0][1].reshape(64,64)
                check2 = utils.renormalize_image(check2)
                check3 = e[1][0][0].reshape(64,64)
                check3 = utils.renormalize_image(check3)
                img = Image.fromarray(check.astype('uint8'), 'L')
                img2 = Image.fromarray(check2.astype('uint8'), 'L')
                img3 = Image.fromarray(check3.astype('uint8'), 'L')
                img.save('../ResNet_Test1/check/input' +str(number) +'.png')
                img2.save('../ResNet_Test1/check/frame' +str(number) +'.png')
                img3.save('../ResNet_Test1/check/test' +str(number) +'.png')
                if len(numpy_x) % 1000 == 0:
                    print("processed %d examples" % len(numpy_x))
            except EOFError:
                print('error')
                break
            except Exception:
                print('error')
                pass
        print("unpickled total %d examples" % len(numpy_x))
    


    # read test data
    # number = 0
    # with gzip.open('../ResNet_Test1/Conpress/Resnet_test.pkl', "rb") as of:
    #     number = number + 1
    #     while number < 10:
    #         try:
    #             e = pickle.load(of)
                
    #             if len(numpy_test) % 1000 == 0:
    #                 print("processed %d examples" % len(numpy_test))
    #         except EOFError:
    #             print('error')
    #             break
    #         except Exception:
    #             print('error')
    #             pass
    #     print("unpickled total %d examples" % len(numpy_test))
    
Second_Package_Data_onehot_Slice_Loder()