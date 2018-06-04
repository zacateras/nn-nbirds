import keras
import os
import sys
from keras.preprocessing.image import ImageDataGenerator
from preprocess import apply, build_ds_meta, bounding_box


def evaluate_model(model_path, test_generator, outfile=None):
    print("Loading model: " + model_path)
    model = keras.models.load_model(model_path)

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                  metrics=['categorical_accuracy', 'top_k_categorical_accuracy'])
    print("Evaluating model...")
    results = model.evaluate_generator(test_generator)

    print(results)
    if outfile is not None:
        print('\t'.join([model_path] + list(map(str, results)) + model.metrics_names), file=outfile)

    del model
    keras.backend.clear_session()


if __name__ == '__main__':

    img_width = 299
    img_height = 299
    batch_size = 32

    data_dir = "data"
    test_dir = os.path.join(data_dir, "SET_A_test")
    bb_test_dir = os.path.join(data_dir, "SET_A_test_BB")

    ds_meta = build_ds_meta()

    apply(bounding_box, test_dir, bb_test_dir, ds_meta)

    datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=0,
        width_shift_range=0,
        height_shift_range=0)

    test_generator = datagen.flow_from_directory(
        bb_test_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        color_mode='rgb')

    models_dir = sys.argv[1]
    outfile_path = sys.argv[2]

    with open(outfile_path, 'w+') as outfile:
        for model in sorted([os.path.join(models_dir, m) for m in os.listdir(models_dir) if m.endswith("hdf5")]):
            evaluate_model(model, test_generator, outfile)
