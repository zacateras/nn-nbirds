import keras

class CustomDnnSvm():
    def __custom_dnn_headless(self, base_model, base_model_top_layer):
        return keras.models.Model(inputs=base_model.input, outputs=base_model.get_layer(top_layer).output)

    def __init__(self, base_model, base_model_top_layer, svm_layer):
        self.base_model = base_model
        self.base_model_headless = self.__custom_dnn_headless(base_model, base_model_top_layer)
        self.svm_layer = svm_layer

    def svm_fit_base_predict(self, generator):
        svm_x = self.base_model_headless.predict_generator(generator, verbose=True)
        svm_y = generator.classes

        return (svm_x, svm_y)

    def svm_fit(self, svm_data):
        train_x, train_y = svm_data

        self.svm_layer.fit(train_x, train_y)

        return self.svm_layer.score(train_x, train_y)

    def predict(self, generator):
        test_x, test_y = self.svm_fit_base_predict(generator)

        return self.svm_fit_base_predict.score(test_x, test_y)
