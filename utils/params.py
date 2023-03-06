#External imports
import os

# ** Script to store global params **

# 4 subfolder, each for 1 type of images
SUB_FOLDERS_ = ['vv', 'vh', 'water_body_label', 'flood_label']

# initial path after download from GCP
TRAIN_FOLDER_PATH = os.path.join(os.path.expanduser('~'), "code", "OriLmd","flood_project","train","train")

# File path end to retrieve images
PATH_END = '*.png'

# 4 types of images
IMAGE_LISTS_ = ['vv', 'vh', 'water_body', 'flood']

# To copy images into new directory data
DESTINATION_FOLDER_PATH = os.path.join(os.path.expanduser('~'), "code","OriLmd", "flood_project","data")
