import keras
import os
import sys
from keras.preprocessing.image import ImageDataGenerator
from preprocess import apply, build_ds_meta, bounding_box


def evaluate_model(model_path, test_generator):
    print("Loading model: " + model_path)
    model = keras.models.load_model(model_path)

    print(model.metrics_names)
    print("Evaluating model...")
    results = model.evaluate_generator(test_generator)
    print(results)


if __name__ == '__main__':

    img_width = 299
    img_height = 299
    batch_size = 32

    data_dir = "data"
    test_dir = os.path.join(data_dir, "SET_A_test")
    bb_test_dir = os.path.join(data_dir, "SET_A_test_BB")

    ds_meta = build_ds_meta()

    #apply(bounding_box, test_dir, bb_test_dir, ds_meta)

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
    for model in sorted([os.path.join(models_dir, m)
                         for m in os.listdir(models_dir) if m.endswith("hdf5")]):
        evaluate_model(model, test_generator)
