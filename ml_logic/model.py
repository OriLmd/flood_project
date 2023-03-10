from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
import os

def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x


def encoder_block(input, num_filters):
    x = conv_block(input, num_filters)
    p = MaxPool2D((2, 2))(x)
    return x, p


def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x


#initialising model
def initialize_unet(input_shape= (256,256,3)):
    inputs = Input(input_shape) #inputs[:][:][-1]

    s1, p1 = encoder_block(inputs, 32)
    s2, p2 = encoder_block(p1, 64)
    s3, p3 = encoder_block(p2, 128)
    s4, p4 = encoder_block(p3, 256)

    b1 = conv_block(p4, 512)

    d1 = decoder_block(b1, s4, 256)
    d2 = decoder_block(d1, s3, 128)
    d3 = decoder_block(d2, s2, 64)
    d4 = decoder_block(d3, s1, 32)

    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4) # la concat en input

    # concat d4 avec input wb
    #concat = Concatenate(axis=-1)([d4,tf.expand_dims(inputs[:,:,:,-1],axis=-1)])
    #outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(concat) # la concat en input


    model = Model(inputs, outputs, name="U-Net")
    return model


def compile_model(model, loss, metric):
    # compile model with specified loss and metric
    model.compile(optimizer='adam', # default learning rate at 0.001
                  loss = loss,
                  metrics=[metric])
    return model


def fit_model(model, train_dataset, val_dataset, drive_folder_path, batch_size= 32, epochs = 100, patience=10):

    es = EarlyStopping(monitor='val_loss', patience = patience, restore_best_weights= True)
    # EarlyStopping(baseline = None) can also be changed;
    # Baseline value for the monitored quantity. Training will stop if the model doesn't show improvement over the baseline.
    model_checkpoint = ModelCheckpoint(filepath = os.path.join(drive_folder_path, "models","checkpoints"),
                                       save_weights_only=True,
                                       save_best_only=True)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0)

    tensorboard = TensorBoard(log_dir=os.path.join(drive_folder_path, "models","logs"),
            histogram_freq=1,
            write_graph=False,
            write_images=False,
            write_steps_per_second=False,
            update_freq='epoch',
            profile_batch=0,
            embeddings_freq=0,
            embeddings_metadata=None)

    history = model.fit(train_dataset.batch(batch_size),
                        callbacks=[es, model_checkpoint, reduce_lr, tensorboard],
                        validation_data = val_dataset.batch(batch_size),
                        epochs = epochs,
                        verbose = 1)

    return (model, history)


if __name__ == "__main__":
    input_shape = (256, 256, 3)
    model = initialize_unet(input_shape)
    model = compile_model(model)
    #train_model(model,)
    print(model.summary())
