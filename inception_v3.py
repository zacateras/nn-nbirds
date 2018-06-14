import keras
import shutil
import os

from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator

from preprocess import apply, build_ds_meta, bounding_box

data_dir = "data"
train_dir = os.path.join(data_dir, "SET_A_train")
validation_dir = os.path.join(data_dir, "SET_A_validation")
bb_train_dir = os.path.join(data_dir, "SET_A_train_BB")
bb_validation_dir = os.path.join(data_dir, "SET_A_validation_BB")

train_tmp_dir = "train_tmp"
validation_tmp_dir = "validation_tmp"

for tmp_dir in [train_tmp_dir, validation_tmp_dir]:
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.mkdir(tmp_dir)

ds_meta = build_ds_meta()

apply(bounding_box, train_dir, bb_train_dir, ds_meta)
apply(bounding_box, validation_dir, bb_validation_dir, ds_meta)

img_width = 299
img_height = 299
batch_size = 32

class_number = len(os.listdir(train_dir))

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    vertical_flip=False,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=0,
    width_shift_range=0,
    height_shift_range=0)

train_generator = train_datagen.flow_from_directory(
    bb_train_dir,
    save_to_dir=train_tmp_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    color_mode='rgb')

validation_generator = validation_datagen.flow_from_directory(
    bb_validation_dir,
    save_to_dir=validation_tmp_dir,
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

model.summary()

# save model after each epoch
checkpointer = keras.callbacks.ModelCheckpoint(filepath='weights-{epoch:02d}.hdf5', verbose=1)

model.fit_generator(train_generator, validation_data=validation_generator, epochs=30, callbacks=[checkpointer])
