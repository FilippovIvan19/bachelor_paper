from tensorflow import keras

from src.models.base_model import BaseModel


class Model_MLP_dl4tsc(BaseModel):
    def __init__(self, input_shape, nb_classes):
        self.callbacks = []
        self.batch_size = 128
        self.nb_epochs = 5000
        # self.nb_epochs = 2

        super().__init__(input_shape, nb_classes)

    def build_model(self, input_shape, nb_classes):
        input_layer = keras.layers.Input(input_shape[::-1])

        # flatten/reshape because when multivariate all should be on the same axis
        input_layer_flattened = keras.layers.Flatten()(input_layer)

        layer_1 = keras.layers.Dropout(0.1)(input_layer_flattened)
        layer_1 = keras.layers.Dense(500, activation='relu')(layer_1)

        layer_2 = keras.layers.Dropout(0.2)(layer_1)
        layer_2 = keras.layers.Dense(500, activation='relu')(layer_2)

        layer_3 = keras.layers.Dropout(0.2)(layer_2)
        layer_3 = keras.layers.Dense(500, activation='relu')(layer_3)

        output_layer = keras.layers.Dropout(0.3)(layer_3)
        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(output_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adadelta(),
                      metrics=[keras.metrics.Recall()])
        return model

    def prepare(self, x_train, y_train, x_test, y_test):
        self.batch_size = int(min(x_train.shape[0] / 10, self.batch_size))
        return x_train.swapaxes(1, 2), y_train, x_test.swapaxes(1, 2), y_test
