#External imports
import cv2
import glob
import shutil
import os

# Internal import
from utils.params import *

# ** Script to load data from local files **

def copy_images_to_folder(folder):
    # Function to retrieve all images from a subfolder (e.g. 'vv')
    # and copy it into the new directory data/subfolder (e.g. 'data/vv')
    source_folder = glob.glob(pathname=os.path.join(TRAIN_FOLDER_PATH, '**', folder,"*.png"), recursive=True)

    # fetch all files
    for file in source_folder:
        filename = file.split(folder+'/')[1]
        destination = os.path.join(DESTINATION_FOLDER_PATH,folder, filename)
        #copy only files
        if os.path.isfile(file):
            shutil.copy(file, destination)
    print('✅ Copied files')
    return None


def load_subfolder_images(main_folder, sub_folder):
    # Function to be called in function load_images
    # Return the list of PNG file names and the list of images
    # From the subfolder located in the main folder
    # main_folder to be declared in notebook
    # sub_folder declared in SUB_FOLDERS_ in params.py

    path = os.path.join(main_folder,sub_folder,PATH_END)

    files_ = sorted([file for file in glob.glob(path)])

    images_ = [cv2.imread(file) for file in files_]

    return files_, images_



def load_folder_images(folder):

    # For each subfolder 'vv', 'vh', 'water_body_label', 'flood_label'
    # 1. Extract file names and images
    # 2. Check first and last file names to check the extraction order
    # as well as the length of files which have been extracted
    # 3. Print corresponding message after assessing the extraction stage


    # Initialisation of list for first and last files
    check_first_ = []
    check_last_ = []

    # Extracting subfolder 'vv'
    f_vv_, im_vv_ = load_subfolder_images(folder, 'vv')

    check_first_.append(f_vv_[0].split('vv')[1])
    check_last_.append(f_vv_[-1].split('vv')[1])

    # Extracting subfolder 'vh'
    f_vh_, im_vh_ = load_subfolder_images(folder, 'vh')
    check_first_.append(f_vh_[0].split('vh')[1])
    check_last_.append(f_vh_[-1].split('vh')[1])

    # Extracting subfolder 'water_body_label'
    f_water_body_, im_water_body_ = load_subfolder_images(folder, 'water_body_label')
    check_first_.append(f_water_body_[0].split('water_body_label')[1].split('.png')[0] + '_')
    check_last_.append(f_water_body_[-1].split('water_body_label')[1].split('.png')[0] + '_')

    # Extracting subfolder 'flood_label'
    f_flood_, im_flood_ = load_subfolder_images(folder, 'flood_label')
    check_first_.append(f_flood_[0].split('flood_label')[1].split('.png')[0] + '_')
    check_last_.append(f_flood_[-1].split('flood_label')[1].split('.png')[0] + '_')

    # Conditional checks on first and last name files and length of extracted lists
    first_file_name = check_first_[0]
    last_file_name = check_last_[0]

    if (check_first_.count(first_file_name) == len(check_first_)) and (check_last_.count(last_file_name) == len(check_last_)):

        if (len(f_vv_) == len(f_vh_)) and (len(f_vh_) == len(f_water_body_)) and (len(f_water_body_) == len(f_flood_)):

            message = f'✅ Correct extraction for folder {folder}'

        else:
            message = '🚨 WARNING: Inconsistent number of files extracted for folder {folder}'

    else:
        message = '🚨 WARNING: Files not extracted in right order for folder {folder}'

    print(message)

    return message, f_vv_, im_vv_, f_vh_, im_vh_, f_water_body_, im_water_body_, f_flood_, im_flood_
