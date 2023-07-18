import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import nltk
import joblib
nltk.download('stopwords')

class RandomForestSpamClassifier:
    def __init__(self):
        self.STOPWORDS = set(stopwords.words('english'))
        self.model = None

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'[^0-9a-zA-Z]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = " ".join(word for word in text.split() if word not in self.STOPWORDS)
        return text

    def train(self, df):
        df['clean_text'] = df['messages'].apply(self.clean_text)
        X = df['clean_text']
        y = df['label']
        model = RandomForestClassifier()
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, shuffle=True, stratify=y)
        self.model = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', model)])
        self.model.fit(x_train, y_train)

    def save_model(self, model_path):
        joblib.dump(self.model, model_path)

    def load_model(self, model_path):
        self.model = joblib.load(model_path)

    def predict(self, message):
        if self.model:
            clean_message = self.clean_text(message)
            prediction = self.model.predict([clean_message])
            return prediction[0]
        else:
            raise Exception("You must train the model before making predictions.")

