from keras import layers, models
import tensorflowjs as tfjs


class ModelGeneration:
    def __init__(self) -> None:
        model = models.Sequential([
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(16, (3, 3), activation='relu'),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dropout(0.2),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(128, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])

        self.model = model

    def compile(self):
        self.model.compile(
            optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        return self.model


class ModelTraining:
    def __init__(self, model):
        self.model = model

    def train(self, X_train, y_train, X_test, y_test, epochs=5):
        self.model.fit(X_train, y_train, validation_data=(
            X_test, y_test), epochs=epochs)

    def save(self):
        self.model.save('../models/tf-model')
        tfjs.converters.save_keras_model(self.model, '../models/tfjs-model')
