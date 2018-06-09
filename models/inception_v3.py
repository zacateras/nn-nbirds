import keras

from keras.layers import Dense, GlobalAveragePooling2D

def inception_v3(width, height, output):
    # import Inception V3 with pretrained weights and without top layer
    base_model = keras.applications.inception_v3.InceptionV3(include_top=False,
                                                             weights='imagenet',
                                                             input_shape=(width, height, 3))
    # set model layers as non trainable
    for layer in base_model.layers:
        layer.trainable = False

    # add untrained layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    x = Dense(1024, activation='relu')(x)
    predictions = Dense(output, activation='softmax')(x)

    model = keras.models.Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    return model
