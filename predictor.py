from model import RandomForestSpamClassifier
from langdetect import detect
import translators as ts
import phonenumbers
import re

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

    def extract_url(self, text):
        url_regex = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        urls = re.findall(url_regex, text)
        return ', '.join(urls) if urls else ''

    def extract_phone_number(self, text):        
        phone_regex = r'\+?\d[\d\-]*\d'
        possible_numbers = re.findall(phone_regex, text)
        valid_numbers = []
        for number in possible_numbers:
            try:
                parsed_number = phonenumbers.parse(number, None)
                if phonenumbers.is_valid_number(parsed_number):
                    valid_numbers.append(number)
            except phonenumbers.phonenumberutil.NumberParseException as e:
                print(str(e))
                try:
                    parsed_number = phonenumbers.parse(number, 'AD')  # Use 'AD' as the default region
                    if phonenumbers.is_valid_number(parsed_number):
                        valid_numbers.append(number)
                except phonenumbers.phonenumberutil.NumberParseException as e:
                    print(str(e))
                    continue
        return ', '.join(valid_numbers) if valid_numbers else ''
    
    def predict(self, message):
        # Escape single quotes                
        message = message.replace("'", "\\'")
        translated_message = self.translate_to_english(message)
        prediction = self.classifier.predict(translated_message)
        malicious_url = self.extract_url(message)
        phone_number = self.extract_phone_number(message)
        return {
            'prediction': prediction,
            'malicious_url': malicious_url,
            'phone_number': phone_number
        }