#External imports
import glob
from tensorflow import data, io, concat

#Internal import


### Script with functions to load and preprocess images ###

def read_one_image(img_path):
    # From 1 image path return an array (256,256,1)
    data = io.read_file(img_path) # read_path
    img = io.decode_png(data, channels=1) # channels=1 to have a grayscale image
    return img

def read_four_images(vv_path, vh_path, wb_path, fl_path):
    # Apply read_one_image to 4 image in parallel
    # To be used within .map on the tf dataset
    vv_img = read_one_image(vv_path)
    vh_img = read_one_image(vh_path)
    wb_img = read_one_image(wb_path)
    fl_img = read_one_image(fl_path)

    return vv_img, vh_img, wb_img, fl_img

def normalize(img):
    # Normalizing image
    return img / 255

def prepare_images(vv_img, vh_img, wb_img, fl_img):
    # Apply normalize to 4 image in parallel
    # To be used within .map on the tf dataset
    vv_img = normalize(vv_img)
    vh_img = normalize(vh_img)
    wb_img = normalize(wb_img)
    fl_img = normalize(fl_img)
    return vv_img, vh_img, wb_img, fl_img

def make_concat(vv_img, vh_img, wb_img, fl_img):
    # concat
    return concat([vv_img,vh_img,wb_img],axis=-1),fl_img
