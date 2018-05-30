import keras
from keras.preprocessing.image import ImageDataGenerator
from preprocess import apply, build_ds_meta, bounding_box
import os
from keras.layers import Dense, GlobalAveragePooling2D

data_dir = "data"
train_dir = os.path.join(data_dir, "SET_A_train")
validation_dir = os.path.join(data_dir, "SET_A_validation")
bb_train_dir = os.path.join(data_dir, "SET_A_train_BB")
bb_validation_dir = os.path.join(data_dir, "SET_A_validation_BB")

ds_meta = build_ds_meta()

# apply(bounding_box, train_dir, bb_train_dir, ds_meta)
# apply(bounding_box, validation_dir, bb_validation_dir, ds_meta)

img_width = 299
img_height = 299
batch_size = 32

class_number = len(os.listdir(train_dir))

datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=0,
    width_shift_range=0,
    height_shift_range=0)

train_generator = datagen.flow_from_directory(
    bb_train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    color_mode='rgb')

validation_generator = datagen.flow_from_directory(
    bb_validation_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    color_mode='rgb')

# import Inception V3 with pretrained weights and without top layer
base_model = keras.applications.inception_v3.InceptionV3(include_top=False,
                                                         weights='imagenet',
                                                         input_shape=(img_height, img_height, 3))
# set model layers as non trainable
for layer in base_model.layers:
    layer.trainable = False

# add untrained layers
x = base_model.output
x = GlobalAveragePooling2D()(x)

x = Dense(1024, activation='relu')(x)
predictions = Dense(class_number, activation='softmax')(x)

model = keras.models.Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['categorical_accuracy'])


checkpointer = keras.callbacks.ModelCheckpoint(filepath='weights-{epoch:02d}.hdf5', verbose=1)

model.fit_generator(train_generator, validation_data=validation_generator, epochs=15, callbacks=[checkpointer])
