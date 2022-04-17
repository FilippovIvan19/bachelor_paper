from tensorflow import keras

from src.models.base_model import BaseModel


class Model_MCDCNN(BaseModel):
    def __init__(self, input_shape, nb_classes):
        self.callbacks = []
        self.batch_size = 16
        # self.nb_epochs = 120
        self.nb_epochs = 7

        super().__init__(input_shape, nb_classes)

    def build_model(self, input_shape, nb_classes):
        n_t = input_shape[0]
        n_vars = input_shape[1]

        padding = 'valid'

        if n_t < 60:  # for ItalyPowerOndemand
            padding = 'same'

        input_layers = []
        conv2_layers = []

        for n_var in range(n_vars):
            input_layer = keras.layers.Input((n_t, 1))
            input_layers.append(input_layer)

            conv1_layer = keras.layers.Conv1D(filters=8, kernel_size=5, activation='relu', padding=padding)(input_layer)
            conv1_layer = keras.layers.MaxPooling1D(pool_size=2)(conv1_layer)

            conv2_layer = keras.layers.Conv1D(filters=8, kernel_size=5, activation='relu', padding=padding)(conv1_layer)
            conv2_layer = keras.layers.MaxPooling1D(pool_size=2)(conv2_layer)
            conv2_layer = keras.layers.Flatten()(conv2_layer)

            conv2_layers.append(conv2_layer)

        if n_vars == 1:
            # to work with univariate time series
            concat_layer = conv2_layers[0]
        else:
            concat_layer = keras.layers.Concatenate(axis=-1)(conv2_layers)

        fully_connected = keras.layers.Dense(units=732, activation='relu')(concat_layer)

        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(fully_connected)

        model = keras.models.Model(inputs=input_layers, outputs=output_layer)

        model.compile(loss='categorical_crossentropy',
                      optimizer=keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, decay=0.0005),
                      metrics=[keras.metrics.Recall()])

        return model

    @staticmethod
    def prepare_input(x):
        new_x = []
        n_t = x.shape[1]
        n_vars = x.shape[2]

        for i in range(n_vars):
            new_x.append(x[:, :, i:i + 1])

        return new_x

    def prepare(self, x_train, y_train, x_test, y_test):
        return self.prepare_input(x_train), y_train, self.prepare_input(x_test), y_test
