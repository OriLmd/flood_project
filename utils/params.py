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

#ensure the path below corresponds to your google drive path
#you can run the following lines to get the cwd
##import os
##os.getcwd()
#the output should replace "/content"
DRIVE_FOLDER_PATH = os.path.join("/content",
                                 "code",
                                 "OriLmd",
                                 "flood_project",
                                 "input-data-for-flood")
