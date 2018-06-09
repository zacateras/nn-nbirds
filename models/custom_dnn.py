import keras

from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPooling2D, Dropout, Flatten

def custom_dnn(width, height, output, descriptor_size=512, filter_size=(3, 3), filter_number=32, kernel_reg=None, kernel_reg_name=None):

    model_code = 'nn_%s_descr_%s_x_%s_%s_filt_%s_batch_%s_reg_%s_epochs' % (descriptor_size,
                        filter_number, filter_size[0], filter_size[1], batch_size, kernel_reg_name, epochs)
        
    model = Sequential()
    model.add(Conv2D(filter_number, filter_size, padding='same', activation='relu', input_shape=(width, height, 3), kernel_regularizer=kernel_reg))
    model.add(Conv2D(filter_number, filter_size, activation='relu', kernel_regularizer=kernel_reg))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(filter_number * 2, filter_size, padding='same', activation='relu', kernel_regularizer=kernel_reg))
    model.add(Conv2D(filter_number * 2, filter_size, activation='relu', kernel_regularizer=kernel_reg))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(filter_number * 2, filter_size, padding='same', activation='relu', kernel_regularizer=kernel_reg))
    model.add(Conv2D(filter_number * 2, filter_size, activation='relu', kernel_regularizer=kernel_reg))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(descriptor_size, activation='relu', kernel_regularizer=kernel_reg))
    model.add(Dropout(0.5))
    model.add(Dense(output, activation='softmax', kernel_regularizer=kernel_reg))

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['categorical_accuracy', 'top_k_categorical_accuracy'])

    return model