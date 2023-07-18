from model import RandomForestSpamClassifier
from langdetect import detect
import translators as ts

class Predictor:
    def __init__(self, model_path):
        self.classifier = RandomForestSpamClassifier()
        self.classifier.load_model(model_path)

    def translate_to_english(self, text):
        if detect(text) != 'en':
            translated_text = ts.translate_text(text)  # translate to English
            return translated_text
        else:
            return text

    def predict(self, message):
        translated_message = self.translate_to_english(message)
        return self.classifier.predict(translated_message)

