"""
Utilities for evaluating svms on neural model outputs

Useful functions:
 * predict_svm_vectors -- for predicting test and training vectors
 * evaluate_svm

"""

import keras
import os
import sys
from keras.preprocessing.image import ImageDataGenerator
from preprocess import apply, build_ds_meta, bounding_box
from sklearn import svm
import pickle


def predict_svm_vectors(directory, base_model, top_layer='dropout_8',
                        image_width=64, image_height=64, batch_size=32):
    """
    Generates tuple of (nn_output_vectors, categories). Top layer is name of new model (one without last layer),
    and must be given by user.

    :param directory:
    :param base_model:
    :param top_layer:
    :param image_width:
    :param image_height:
    :param batch_size:
    :return:
    """
    model = keras.models.Model(inputs=base_model.input, outputs=base_model.get_layer(top_layer).output)

    generator = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=0,
        width_shift_range=0,
        height_shift_range=0,
        vertical_flip=False,
        horizontal_flip=False
    ).flow_from_directory(
        directory,
        target_size=(image_width, image_height),
        batch_size=batch_size,
        color_mode='rgb')

    svm_x = model.predict_generator(generator, verbose=True)
    svm_y = generator.classes

    return svm_x, svm_y


def evaluate_svm(test_data, train_data, classifier, logfile=None):
    """
    Evaluates svm, writes output to logfile in tsv format with columns:
    - svm description
    - accuracy on test set
    - accuracy on train set

    """
    train_x, train_y = train_data
    classifier.fit(train_x, train_y)

    test_x, test_y = test_data

    train_accuracy = classifier.score(train_x, train_y)
    test_accuracy = classifier.score(test_x, test_y)

    classifier_description=str(classifier).replace('\n', ' ')
    out_msg = '\t'.join((classifier_description, str(test_accuracy), str(train_accuracy)))
    print(out_msg)

    if logfile is not None:
        with open(logfile, 'a+') as lf:
            lf.writelines([out_msg])

    return classifier_description, test_accuracy, train_accuracy

def evaluate_nn_model(directory, model,
                      image_width=64, image_height=64, batch_size=32):
    generator = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=0,
        width_shift_range=0,
        height_shift_range=0,
        vertical_flip=False,
        horizontal_flip=False
    ).flow_from_directory(
        directory,
        target_size=(image_width, image_height),
        batch_size=batch_size,
        color_mode='rgb')

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['categorical_accuracy'])

    accuracy = model.evaluate_generator(generator)

    out_msg = '\t'.join(('nn_categorical_accuracy', (str(accuracy))))
    print(out_msg)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        base_model = keras.models.load_model(sys.argv[1])

        data_dir = "data"
        train_dir = os.path.join(data_dir, "SET_A_train")
        test_dir = os.path.join(data_dir, "SET_A_test")
        bb_train_dir = os.path.join(data_dir, "SET_A_train_BB")
        bb_test_dir = os.path.join(data_dir, "SET_A_test_BB")

        ds_meta = build_ds_meta()

        # apply(bounding_box, train_dir, bb_train_dir, ds_meta)
        # apply(bounding_box, test_dir, bb_test_dir, ds_meta)

        test_data = predict_svm_vectors(bb_test_dir, base_model)
        train_data = predict_svm_vectors(bb_train_dir, base_model)

        evaluate_nn_model(bb_test_dir, base_model)

        with open('svm_tmp_data.pkl', 'wb') as data_tmp_file:
            pickle.dump(test_data, data_tmp_file, pickle.HIGHEST_PROTOCOL)
            pickle.dump(train_data, data_tmp_file, pickle.HIGHEST_PROTOCOL)

    else:
        # if there is no model, load data from tmp file
        with open('svm_tmp_data.pkl', 'rb') as data_tmp_file:
            test_data = pickle.load(data_tmp_file)
            train_data = pickle.load(data_tmp_file)

    svm_classifier = svm.SVC()

    evaluate_svm(test_data, train_data, svm_classifier, 'log.tsv')
