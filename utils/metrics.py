from keras import backend as K
import numpy as np
import tensorflow as tf

#recover y_true from dataset
def get_ytrue_for_prediction(dataset):
    # batch the dataset into a fixed batch size
    batch_size = 32
    batched_dataset = dataset.batch(batch_size)
    # iterate over the prefetched dataset and convert it into a TensorFlow tensor
    for batch in batched_dataset.as_numpy_iterator():
        y_pred = batch[2]  # get the wb_img in the 3rd postion and fl_img int the 4th from prefetched_dataset
        y_pred_tensor = tf.convert_to_tensor(y_pred)
    return y_pred_tensor

#recover y_pred from dataset
def get_ypred_for_prediction(dataset):
    # batch the dataset into a fixed batch size
    batch_size = 32
    batched_dataset = dataset.batch(batch_size)
    # iterate over the prefetched dataset and convert it into a TensorFlow tensor
    for batch in batched_dataset.as_numpy_iterator():
        y_true = batch[3]  # get the wb_img in the 3rd postion and fl_img int the 4th from prefetched_dataset
        y_true_tensor = tf.convert_to_tensor(y_true)
    return y_true_tensor

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
    def reset_states(self):
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
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.int32)
        false_neg = y_pred - y_true
        false_pos = y_true - y_pred
        self.FN.assign_add(tf.reduce_sum(tf.cast(false_neg == -1, tf.float32)) / (256*256))
        self.FP.assign_add(tf.reduce_sum(tf.cast(false_pos == -1, tf.float32)) / (256*256))
    def result(self):
        return self.FN + self.FP
    def reset_states(self):
        self.FN.assign(0.0)
        self.FP.assign(0.0)

# Class to use dice as loss in model
class DiceLoss(tf.keras.losses.Loss):

    def __init__(self, name='dice_loss'):
        super().__init__(name=name)

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
    y_true=get_ytrue_for_prediction(dataset)
    y_pred=get_ypred_for_prediction(dataset)
    baseline_score_dice=dice(y_true, y_pred)
    baseline_score_totalerror=total_error(y_true, y_pred)
    return f'Dice_coef score is {baseline_score_dice.numpy()} and total error score is {baseline_score_totalerror}'
