import keras


class DataPreProcessing:
    def __init__(self) -> None:
        pass

    def load_data(self):
        (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def normalize(self):
        self.X_train /= 255.0
        self.X_test /= 255.0

        self.y_train = keras.utils.to_categorical(self.y_train)
        self.y_test = keras.utils.to_categorical(self.y_test)

    def get_data(self):
        return (self.X_train, self.y_train), (self.X_test, self.y_test)
