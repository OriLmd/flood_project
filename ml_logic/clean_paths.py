import cv2
import csv
import numpy as np

def clean_list(path_list, percentage_black = 30, percentage_white = 10):
    '''This function is used to return a list of paths of images who has fewer
    than percentage black/white of pixels.
    percentages must be entered as a whole number, not a fraction (ex. 50% = 50 not 0.5).
    Returns a list of paths of images that have the right paths'''
    t_hold_b = 65536*(percentage_black/100)
    #t_hold_black = 0 + t_hold_b
    t_hold_w = 65536*(percentage_white/100)
    #t_hold_white = 65536 - t_hold_w
    new_path = []
    for path in path_list:
        img_gs = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_gs_norm = img_gs/255
        im_nor_w = img_gs_norm[img_gs_norm == 1]
        im_nor_b = img_gs_norm[img_gs_norm == 0]
        pixels_w = len(im_nor_w)
        pixels_b = len(im_nor_b)
        #pixel_sum = np.sum(img_gs_norm)
        if pixels_w < t_hold_w and pixels_b < t_hold_b:
            new_path.append(path)
    return new_path

path_vvs_not_clean = glob.glob('train/train/*/tiles/vv/*_vv.png')
path_vvs = clean_list(path_vvs_not_clean, percentage_black=10, percentage_white=10)
path_vhs = [file.split('vv')[0] + 'vh' + file.split('vv')[1] + 'vh' + '.png' for file in path_vvs]
path_wbs = [(file.split('vv')[0] + 'water_body_label' + file.split('vv')[1]).strip('_') + '.png' for file in path_vvs]
path_fls = [(file.split('vv')[0] + 'flood_label' + file.split('vv')[1]).strip('_') + '.png' for file in path_vvs]

# open a new csv file for writing
with open('paths_cleaned_images.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    # write the header row
    writer.writerow(['vv_path', 'vh_path', 'wb_path', 'fl_path'])
    # write the data rows
    for i in range(len(path_vvs)):
        writer.writerow([path_vvs[i], path_vhs[i], path_wbs[i], path_fls[i]])
