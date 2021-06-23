from unet_model import *
from gen_patches import *

import os.path
import numpy as np
import tifffile as tiff
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint


def normalize(img):
    min = img.min()
    max = img.max()
    x = 2.0 * (img - min) / (max - min) - 1.0
    return x


def rotate_img(img, r):
    # channels in img are last!!!
    # r is a transformation type (an integer from 0 to 7)
    reverse_x = r % 2 == 1         # True for r in [1,3,5,7]
    reverse_y = (r // 2) % 2 == 1  # True for r in [2,3,6,7]
    swap_xy = (r // 4) % 2 == 1    # True for r in [4,5,6,7]
    if reverse_x:
        img = img[::-1, :, :]
    if reverse_y:
        img = img[:, ::-1, :]
    if swap_xy:
        img = img.transpose([1, 0, 2])
    return img





N_BANDS = 8
N_CLASSES = 5  # buildings, roads, trees, crops and water
CLASS_WEIGHTS = [0.2, 0.3, 0.1, 0.1, 0.3]
N_EPOCHS = 100
UPCONV = True
PATCH_SZ = 160   # should divide by 16
BATCH_SIZE = 100
TRAIN_SZ = 4000  # train size
VAL_SZ = 1000    # validation size


def get_model():
    return unet_model(N_CLASSES, PATCH_SZ, n_channels=N_BANDS, upconv=UPCONV, class_weights=CLASS_WEIGHTS)


weights_path = 'weights'
if not os.path.exists(weights_path):
    os.makedirs(weights_path)
weights_path += '/unet_weights.hdf5'

trainIds = [str(i).zfill(2) for i in range(1, 25)]  # all availiable ids: from "01" to "24"


if __name__ == '__main__':
    X_DICT_TRAIN = dict()
    Y_DICT_TRAIN = dict()
    X_DICT_VALIDATION = dict()
    Y_DICT_VALIDATION = dict()

    counter = 0
    print('Reading images')
    for img_id in trainIds:
        for i in range(8):
            img_m = normalize(tiff.imread('./data/mband/{}.tif'.format(img_id)).transpose([1, 2, 0]))
            mask = tiff.imread('./data/gt_mband/{}.tif'.format(img_id)).transpose([1, 2, 0]) / 255
            print(img_m.shape)
            print(mask.shape)
            img_m = rotate_img(img_m,i)
            mask = rotate_img(mask,i)
            print(img_m.shape)
            print(mask.shape)
            train_xsz = int(3/4 * img_m.shape[0])  # use 75% of image as train and 25% for validation
            X_DICT_TRAIN[counter] = img_m[:train_xsz, :, :]
            Y_DICT_TRAIN[counter] = mask[:train_xsz, :, :]
            X_DICT_VALIDATION[counter] = img_m[train_xsz:, :, :]
            Y_DICT_VALIDATION[counter] = mask[train_xsz:, :, :]
            print("{} read.".format(counter))
            counter += 1

    print('Images were read')

    def train_net():
        print("start train net")
        x_train, y_train = get_patches(X_DICT_TRAIN, Y_DICT_TRAIN, n_patches=TRAIN_SZ, sz=PATCH_SZ)
        x_val, y_val = get_patches(X_DICT_VALIDATION, Y_DICT_VALIDATION, n_patches=VAL_SZ, sz=PATCH_SZ)
        model = get_model()
        if os.path.isfile(weights_path):
            model.load_weights(weights_path)
        model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', save_best_only=True)
        csv_logger = CSVLogger('log_unet.csv', append=True, separator=';')
        tensorboard = TensorBoard(log_dir='./tensorboard_unet/', write_graph=True, write_images=True)
        model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS,
                  verbose=2, shuffle=True,
                  callbacks=[model_checkpoint, csv_logger, tensorboard],
                  validation_data=(x_val, y_val))
        return model

    train_net()
