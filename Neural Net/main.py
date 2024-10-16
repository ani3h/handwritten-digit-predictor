from DataPreProcessing import DataPreProcessing
from Model import ModelGeneration, ModelTraining


# -----GLOBAL------
EPOCHS = 10


if __name__ == '__main__':
    dpp = DataPreProcessing()
    dpp.load_data()
    dpp.normalize()

    model = ModelGeneration().compile()

    trainer = ModelTraining(model)
    trainer.train(*dpp.get_data()[0], *dpp.get_data()[-1], EPOCHS)
    trainer.save()
