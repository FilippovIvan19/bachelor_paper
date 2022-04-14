from tensorflow import keras
# from src.utils.classifiers_utils import save_logs, calculate_metrics
from src.models.base_model import BaseModel


class Model_CNN(BaseModel):

    # def __init__(self, output_directory, input_shape, nb_classes):
    #     super().__init__(output_directory, input_shape, nb_classes)
    #     self.output_directory = output_directory
    #
    #     self.model.save_weights(self.output_directory + 'model_init.hdf5')
    #
    #     return

    def build_model(self, input_shape, nb_classes):
        self.batch_size = 16
        self.nb_epochs = 100
        # self.nb_epochs = 2000

        padding = 'valid'
        input_layer = keras.layers.Input(input_shape)

        if input_shape[0] < 60: # for italypowerondemand dataset
            padding = 'same'

        conv1 = keras.layers.Conv1D(filters=6,kernel_size=7,padding=padding,activation='sigmoid')(input_layer)
        conv1 = keras.layers.AveragePooling1D(pool_size=3)(conv1)

        conv2 = keras.layers.Conv1D(filters=12,kernel_size=7,padding=padding,activation='sigmoid')(conv1)
        conv2 = keras.layers.AveragePooling1D(pool_size=3)(conv2)

        flatten_layer = keras.layers.Flatten()(conv2)

        output_layer = keras.layers.Dense(units=nb_classes,activation='sigmoid')(flatten_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(),
                      metrics=[keras.metrics.Recall()])
        #
        # file_path = self.output_directory + 'best_model.hdf5'
        #
        # model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
        #                                                    save_best_only=True)
        #
        # self.callbacks = [model_checkpoint]

        return model

    # def fit(self, x_train, y_train, x_val, y_val, y_true):
    #
    #     hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
    #                           verbose=self.verbose, validation_data=(x_val, y_val), callbacks=self.callbacks)
    #
    #     self.model.save(self.output_directory+'last_model.hdf5')
    #
    #     model = keras.models.load_model(self.output_directory + 'best_model.hdf5')
    #     save_logs(self.output_directory, hist, y_pred, y_true, duration,lr=False)

    # def predict(self, x_test,y_true,x_train,y_train,y_test,return_df_metrics = True):
    #     model_path = self.output_directory + 'best_model.hdf5'
    #     model = keras.models.load_model(model_path)
    #     y_pred = model.predict(x_test)
    #     if return_df_metrics:
    #         y_pred = np.argmax(y_pred, axis=1)
    #         df_metrics = calculate_metrics(y_true, y_pred, 0.0)
    #         return df_metrics
    #     else:
    #         return y_pred

    # def prepare(self, x_train, y_train, x_test, y_test):
    #     pass
