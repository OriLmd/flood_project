from keras import backend as K
import numpy as np

#compute IoU
def iou_coef(y_true, y_pred, smooth=1):
  intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
  union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
  iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
  return iou

#compute Dice coef
def dice_coef(y_true, y_pred, smooth=1):
  intersection = K.sum(y_true * y_pred, axis=[1,2,3])
  union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
  dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
  return dice

#compute Total Error
def total_error(y_true, y_pred):
    FN=[]
    FP=[]
    for image_true,image_pred in zip(y_true,y_pred):
        #for every images, transform into int32 (for substraction)
        y_true_int32=image_true.astype('int32')
        y_pred_int32=image_pred.astype('int32')

        #compute False negative and Falso positive
        false_neg=y_pred_int32-y_true_int32
        false_pos=y_true_int32-y_pred_int32
        FN.append(np.sum(false_neg==-255))
        FP.append(np.sum(false_pos==-255))

        #sum all the errors
        total_error=FN+FP
    return sum(total_error)
