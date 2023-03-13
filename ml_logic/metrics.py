import numpy as np
import tensorflow as tf
from ml_logic.results import split_tensor_channel
from tensorflow_addons.metrics import FBetaScore
import tensorflow_datasets as tfds


from ml_logic import load_preprocess

#get y_true from dataset
def extract_flood_from_all(vv,vh,wb,flood):
    return flood

def get_ytrue(dataset):
    all_floods = []
    for flood in tfds.as_numpy(dataset.map(load_preprocess.read_four_images).map(load_preprocess.prepare_images).map(extract_flood_from_all)):
        all_floods.append(flood)
    all_floods = np.array(all_floods)
    return all_floods


#get y_pred from dataset
def extract_wb_from_all(vv,vh,wb,flood):
    return wb

def get_ypred_baseline(dataset):
    all_wb = []
    for wb in tfds.as_numpy(dataset.map(load_preprocess.read_four_images).map(load_preprocess.prepare_images).map(extract_wb_from_all)):
        all_wb.append(wb)
    all_wb = np.array(all_wb)
    return all_wb


# Class to use dice as metrics in model
class Dice(tf.keras.metrics.Metric):
    def __init__(self, name='dice', **kwargs):
        super(Dice, self).__init__(name=name, **kwargs)
        self.num_classes = 2 # 2 because with 2 categories water (1) or land (0)
        self.intersection = self.add_weight(name='intersection', initializer='zeros')
        self.union = self.add_weight(name='union', initializer='zeros')
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Reshape the inputs to have a shape of (batch_size * num_pixels, num_classes)
        y_true = tf.reshape(y_true, [-1, self.num_classes])
        y_pred = tf.reshape(y_pred, [-1, self.num_classes])
        # Convert the one-hot encoded y_true and y_pred tensors into class indices
        y_true = tf.cast(tf.argmax(y_true, axis=-1), tf.float32)
        y_pred = tf.cast(tf.argmax(y_pred, axis=-1), tf.float32)
        # Dice coefficient is computed for each class by summing over the true positive, false positive, and false negative counts.
        intersection = tf.reduce_sum(y_true * y_pred, axis=0)
        union = tf.reduce_sum(y_true, axis=0) + tf.reduce_sum(y_pred, axis=0)
        self.intersection.assign_add(intersection)
        self.union.assign_add(union)
    def result(self):
        # Dice coefficient is averaged over all classes and returned as the final metric value
        epsilon = 1e-7
        dice = (2.0 * self.intersection + epsilon) / (self.union + epsilon)
        return tf.reduce_mean(dice)
    def reset_state(self):
        # Reset the internal state of the metric between epochs during training
        self.intersection.assign(0)
        self.union.assign(0)

# Class to use totalerror as metrics in model
class TotalError(tf.keras.metrics.Metric):
    def __init__(self, name='total_error', **kwargs):
        super(TotalError, self).__init__(name=name, **kwargs)
        self.FN = self.add_weight(name='false_negatives', initializer='zeros')
        self.FP = self.add_weight(name='false_positives', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        #compute false negative and false postive sum for each image
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.int32)
        false_neg = y_pred - y_true
        false_pos = y_true - y_pred
        self.FN.assign_add(tf.reduce_sum(tf.cast(false_neg == -1, tf.float32)) )
        self.FP.assign_add(tf.reduce_sum(tf.cast(false_pos == -1, tf.float32)) )

    def result(self):
        # sum of total error over all dataset

        total_errors = self.FN + self.FP
        # all_y=tf.shape(y_true)[0]
        return total_errors


    def reset_state(self):
        #reset value after eache epoch
        self.FN.assign(0.0)
        self.FP.assign(0.0)

# Class to use dice as loss in model
class DiceLoss(tf.keras.losses.Loss):

    def __init__(self, name='dice_loss', reduction=tf.keras.losses.Reduction.AUTO):
        super().__init__(name=name)
        super().__init__(reduction=reduction)

    def call(self, y_true, y_pred, smooth=1e-6):
        numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3)) + smooth
        denominator = tf.reduce_sum(y_true + y_pred, axis=(1, 2, 3)) + smooth
        coefficient = numerator / denominator
        loss = 1 - coefficient
        return tf.reduce_mean(loss)

#get baseline score
def get_baseline_score(dataset):
    dice=Dice()
    total_error=TotalError()
    f2score=FBetaScore(num_classes=1, beta=2.0, threshold=0.5, average='micro')
    y_true=get_ytrue(dataset)
    y_pred=get_ypred_baseline(dataset)
    baseline_score_dice=dice(y_true, y_pred)
    baseline_score_totalerror=total_error(y_true, y_pred)
    baseline_score_f2score=f2score(y_true, y_pred)
    return {'baseline_score_dice': baseline_score_dice.numpy(), 'baseline_score_totalerror':round(baseline_score_totalerror.numpy(),3),'baseline_score_F2score':round(baseline_score_f2score.numpy(),3)}
