import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import nltk
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

    def predict(self, message):
        if self.model:
            clean_message = self.clean_text(message)
            prediction = self.model.predict([clean_message])
            return prediction[0]
        else:
            raise Exception("You must train the model before making predictions.")

# Usage
classifier = RandomForestSpamClassifier()
df = pd.read_csv('datasets/spam_randomForest.csv', encoding='latin-1')
df = df[['v2', 'v1']]
df = df.rename(columns={'v2': 'messages', 'v1': 'label'})
classifier.train(df)

# Smishing messages
print(classifier.predict("Congratulations! You've won a $1,000 gift card. Click here to claim now: www.bit.ly/12345"))
print(classifier.predict("Your bank account has been compromised. Please confirm your details at www.fakebank.com to secure it."))
print(classifier.predict("You've been selected for a free vacation! Call now at 1-800-123-4567 to claim."))
print(classifier.predict("You've won the lottery! Click here to claim your prize: www.lottery.com"))
print(classifier.predict("You have a tax refund due! Click here to provide your bank details for the refund: www.irsscam.com"))
print(classifier.predict("This is your bank. We need you to confirm your security details at www.phishybank.com"))
print(classifier.predict("Free entry in 2 a weekly competition. Text WIN to 80085 to receive entry."))
print(classifier.predict("You have been chosen to receive a free iPhone. Just go to www.getyourfreeiphone.com"))
print(classifier.predict("Your Netflix account is about to be suspended. Click here to update your payment details: www.netflixphish.com"))
print(classifier.predict("Your email account has been hacked. Please go to www.emailrecoveryscam.com to secure it."))

# Ham messages
print(classifier.predict("Hey, are we still on for dinner tonight?"))
print(classifier.predict("Don't forget to pick up the kids from school today."))
print(classifier.predict("I left my keys at your place. Can you check?"))
print(classifier.predict("Your appointment is scheduled for tomorrow at 10 AM."))
print(classifier.predict("Can you call me when you get a chance?"))
print(classifier.predict("Just wanted to say I had a great time last night. We should do it again sometime."))
print(classifier.predict("Your prescription is ready for pickup."))
print(classifier.predict("Can you send me the recipe for that cake you made?"))
print(classifier.predict("Just checking in. How have you been?"))
print(classifier.predict("Your car is ready for pickup at the garage."))
