import abc


class BaseModel(abc.ABC):
    def __init__(self, input_shape, nb_classes):
        self.batch_size = 16
        self.nb_epochs = 3
        self.model = self.build_model(input_shape, nb_classes)

    @abc.abstractmethod
    def build_model(self, input_shape, nb_classes):
        return

    def fit(self, x_train, y_train, x_test, y_test):
        return

    def prepare(self, x_train, y_train, x_test, y_test):
        pass

    def predict(self, x):
        return self.model.predict(x)