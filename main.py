#External imports
import glob
from tensorflow import data


#Internal import
from ml_logic.load_preprocess import read_four_images, prepare_images

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
    return dataset

if __name__ == '__main__':
    dataset = create_dataset()
    small_dataset = dataset.take(32)
    small_dataset = load_and_preprocess_data(small_dataset)
    for pair in small_dataset.take(1):
        print(pair)
