from tensorflow import keras
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from keras.models import Model
from keras.callbacks import EarlyStopping

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
    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, 32)
    s2, p2 = encoder_block(p1, 64)
    s3, p3 = encoder_block(p2, 128)
    s4, p4 = encoder_block(p3, 256)

    b1 = conv_block(p4, 512)

    d1 = decoder_block(b1, s4, 256)
    d2 = decoder_block(d1, s3, 128)
    d3 = decoder_block(d2, s2, 64)
    d4 = decoder_block(d3, s1, 32)

    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    model = Model(inputs, outputs, name="U-Net")
    return model


def compile_model(model, loss = 'mse'):

    model.compile(optimizer='adam',
                  loss = loss,
                  metrics=['binary_accuracy'])
    return model

#def train_model(model, dataset, batch_size = 32, epochs = 5, validation_split = 0.2):
#    history = model.fit(dataset.batch(batch_size), epochs = epochs, validation_split = validation_split)
#    return model, history
#


#replacing function train_model with function fit_model
#function fit_model follows le wagon format and fits model on dataset


def fit_model(model, dataset, batch_size= 32, epochs = 5, validation_split = 0.2, patience=3):

    es = EarlyStopping(monitor='val_loss', patience = patience, restore_best_weights= True)
    # EarlyStopping(baseline = None) can also be changed;
    # Baseline value for the monitored quantity. Training will stop if the model doesn't show improvement over the baseline.


    model_hist = model.fit(dataset,
                        callbacks=[es],
                        validation_split= validation_split,
                        batch_size= batch_size,
                        epochs = epochs,
                        verbose = 1)
    return (model, model_hist)


if __name__ == "__main__":
    input_shape = (256, 256, 3)
    model = initialize_unet(input_shape)
    model = compile_model(model)
    #train_model(model,)
    print(model.summary())
