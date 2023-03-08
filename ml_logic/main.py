#External imports
import glob
from tensorflow import data
import tensorflow as tf
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.utils import split_dataset
from tensorflow_addons.losses import SigmoidFocalCrossEntropy
from tensorflow.keras.metrics import MeanIoU

from tensorflow.keras.losses import Reduction
import numpy as np


#Internal import
from ml_logic.load_preprocess import read_four_images, prepare_images, make_concat
from ml_logic.model import initialize_unet, compile_model, fit_model

### Main script ###

def create_dataset():
    # to create a tf dataset with our images

    # 1. create the lists of paths for each image type
    path_vvs = glob.glob('train/train/*/tiles/vv/*_vv.png')
    path_vhs = [file.split('vv')[0] + 'vh' + file.split('vv')[1] + 'vh' + '.png' for file in path_vvs]
    path_wbs = [(file.split('vv')[0] + 'water_body_label' + file.split('vv')[1]).strip('_') + '.png' for file in path_vvs]
    path_fls = [(file.split('vv')[0] + 'flood_label' + file.split('vv')[1]).strip('_') + '.png' for file in path_vvs]

    # 2. create the tf dataset
    dataset = data.Dataset.from_tensor_slices((path_vvs, path_vhs, path_wbs, path_fls))

    return dataset

def load_and_preprocess_data(dataset): # Warning: dataset must be a batch - not the whole dataset
    # Apply line by line, the methods read_four_images and prepare_images
    dataset = dataset.map(read_four_images) # return 4 tensor arrays (256,256,1)
    dataset = dataset.map(prepare_images) # /255 the 4 tensor arrays
    dataset = dataset.map(make_concat)
    return dataset


def train_test_split(dataset, train_size = 0.6, test_size = 0.2, val_size = 0.2, seed=None):
    #splits a tensor dataset into train,test an val
    #first split into train and both = (test+val)
    #then split both into test and val
    both_size = test_size + val_size
    dataset_train, dataset_both = split_dataset(dataset,left_size=train_size,
                                      right_size=both_size,
                                      shuffle=True,seed=seed)
    dataset_test, dataset_val = split_dataset(dataset_both,left_size=test_size,
                                      right_size=val_size)
    return dataset_train, dataset_test, dataset_val

def train_model(train_dataset, val_dataset,metric='MeanIoU(num_classes=2)',loss='SigmoidFocalCrossEntropy(reduction = Reduction.NONE)',batch_size=16, epochs=5, patience=1):
    # Method to run model

    # 1. Initialize model
    input_shape= (256,256,3)
    model = initialize_unet(input_shape)

    # 2. Compile model - to be updated with loss and metric (e.g. Dice)
    model = compile_model(model, loss, metric)
    model, history = fit_model(model, train_dataset, val_dataset, batch_size=batch_size, epochs=epochs, patience=patience)

    val_loss = np.min(history.history['val_loss'])

    val_metric_key = list(history.history.keys())[-1]
    val_metric = np.min(history.history[val_metric_key])

    #AJOUTER SAVE MODEL AND SAVE RESULTS

    print("✅ train() done \n")
    return val_loss, val_metric

if __name__ == '__main__':
    dataset = create_dataset() # load or dataset with image paths
    small_dataset = dataset.take(32) # take a batch - à creuser
    small_dataset = load_and_preprocess_data(small_dataset) # load and preproc batch into arrays
    small_train, small_test = train_test_split(small_dataset)
    for pair in small_dataset.take(1):
        print(pair)
    for this in small_train.take(1):
        print(this)
