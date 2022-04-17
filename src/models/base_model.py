import abc


class BaseModel(abc.ABC):
    def __init__(self, input_shape, nb_classes):
        self.model = self.build_model(input_shape, nb_classes)

    @abc.abstractmethod
    def build_model(self, input_shape, nb_classes):
        return

    def fit(self, x_train, y_train, x_test, y_test):
        return

    def prepare(self, x_train, y_train, x_test, y_test):
        return x_train, y_train, x_test, y_test

    def predict(self, x):
        return self.model.predict(x)
