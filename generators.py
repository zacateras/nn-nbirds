class Generators:
    def __init__(self, width, height, batch_size):
        self.width = width
        self.height = height
        self.batch_size = batch_size

    def train(self, path='data/SET_A_RES_train'):
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            vertical_flip=False,
            horizontal_flip=True)

        train_generator = train_datagen.flow_from_directory(
            path,
            save_to_dir=train_tmp_dir,
            target_size=(self.width, self.height),
            batch_size=self.batch_size,
            color_mode='rgb')

        count_classes = len(os.listdir(path))

        return (train_generator, count_classes)

    def validation(self, path='data/SET_A_RES_validation'):
        return self.test(path)
    
    def test(self, path='data/SET_A_RES_test'):
        test_datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=0,
            width_shift_range=0,
            height_shift_range=0)

        test_generator = validation_datagen.flow_from_directory(
            path,
            save_to_dir=validation_tmp_dir,
            target_size=(self.width, self.height),
            batch_size=self.batch_size,
            color_mode='rgb')

        count_classes = len(os.listdir(path))

        return (test_generator, count_classes)
