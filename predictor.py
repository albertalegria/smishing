from model import RandomForestSpamClassifier

class Predictor:
    def __init__(self, model_path):
        self.classifier = RandomForestSpamClassifier()
        self.classifier.load_model(model_path)

    def predict(self, message):
        return self.classifier.predict(message)
