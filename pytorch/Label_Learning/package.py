import argparse
import glob
import os
import pickle
import random
import numpy as np
from PIL import Image
import scipy.misc as misc
import numpy as np
import io
import gzip

def normalize_image(img):
    normalized = img/255
    return normalized

def make_one_hot() :
    a = np.array([a for a in range(2350)])
    return a
# using Copress data
# Conpress data is amount 20
# read train, test data
def pickle_slice_examples(package_dir):
    #read directory
    train_dir = '../../../hhjung/Conpress_Son/label/'
    #read all handwrite data
    file_list = os.listdir(train_dir)
    random.shuffle(file_list)
    train_datas = file_list[:90]
    test_datas = file_list[90:]
    #read each handwrite data
    train_data_name = 'font_train_label.pkl'
    test_data_name = 'font_test_label.pkl'
    #make data
    make_data_process(package_dir, train_data_name, train_datas, train_dir)
    make_data_process(package_dir, test_data_name, test_datas, train_dir)


def make_data_process(package_dir, train_data_name, train_datas, train_dir):
    with gzip.open(package_dir + train_data_name, 'wb') as ft:
        for file in train_datas:
            train_paths = glob.glob(os.path.join(train_dir + file, '*.jpg'))
            X_data = []
            label_data = make_one_hot()
            for train_iter in train_paths:
                train_image = np.array(Image.open(train_iter))
                train_image = normalize_image(train_image)
                train_image = train_image.reshape(1, 64, 64)
                X_data.append(train_image)
            X_data = np.array(X_data)
            if (2350, 1, 64, 64) == X_data.shape:
                print(label_data)
                example = (X_data, label_data)
                pickle.dump(example, ft)
        ft.close()

if __name__ == "__main__":
    package_dir = './Conpress/'
    print('start')
    pickle_slice_examples(package_dir)
    
