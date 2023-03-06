#External imports
import numpy as np
import matplotlib.pyplot as plt

#Internal import
from utils.params import *

### Script with basic functions to explore data ###

def get_image_basic_info(images_dict):
    # For the 4 list of images stored in images_dict
    # The function will execute the sub_function 'explore_subset_images'

    def explore_subset_images(image_list_name, image_list_):
        # For the list of images:
        # 1. Print the shape
        # 2. Print whether the images must be normalized
        # 3. Plot some images

        # Explored images:
        print(f'🔍 Information on {image_list_name} images:')

        # Shape of images
        print(f'Shape of 1 image is: {image_list_[0].shape}')

        # Need to normalize
        if np.unique(image_list_[51]).max() > 1 and np.unique(image_list_[51]).min() >= 0:
            print(f'🚨 Images need to be normalized between 0 and 1. Current pixels are between 0 and 255.')
        elif np.unique(image_list_[51]).max() <= 1 and np.unique(image_list_[51]).min() >= 0:
            print('✅ Images already normalized: pixels are between 0 and 1.')
        else:
            print('🚨 More investigation on pixel is required.')

        # Examples of images
        plt.figure(figsize=(20,3))
        plt.suptitle(f'Examples of {image_list_name} images')
        for i in range(8):
            plt.subplot(1,8,i+1)
            plt.imshow(image_list_[50+i])
        plt.show()

    for image_list_name, image_list_ in images_dict.items():
        explore_subset_images(image_list_name, image_list_)
