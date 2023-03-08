#External imports
import glob
from tensorflow import data, keras
from keras.layers import Concatenate
from tensorflow import keras
from keras.utils import split_dataset
from tensorflow import size


#Internal import
from ml_logic.load_preprocess import read_four_images, prepare_images, make_concat


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


def train_test_split(dataset, train_size = 0.8, test_size = 0.2):
    #splits a tensor dataset into train and test (validation happens in the model)
    dataset_train, dataset_test = split_dataset(dataset,left_size=train_size,
                                      right_size=test_size,
                                      shuffle=True,seed=18)
    return dataset_train, dataset_test



if __name__ == '__main__':
    dataset = create_dataset() # load or dataset with image paths
    small_dataset = dataset.take(32) # take a batch - à creuser
    small_dataset = load_and_preprocess_data(small_dataset) # load and preproc batch into arrays
    small_train, small_test = train_test_split(small_dataset)
    for pair in small_dataset.take(1):
        print(pair)
    for this in small_train.take(1):
        print(this)
