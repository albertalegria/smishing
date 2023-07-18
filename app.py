from flask import Flask, request, jsonify
from model import RandomForestSpamClassifier
from predictor import Predictor
import pandas as pd

app = Flask(__name__)
model_path = 'models/model.pkl'
classifier = RandomForestSpamClassifier()

@app.route('/train', methods=['GET'])
def train():
    df = pd.read_csv('datasets/spam_randomForest.csv', encoding='latin-1')
    df = df[['v2', 'v1']]
    df = df.rename(columns={'v2': 'messages', 'v1': 'label'})
    classifier.train(df)
    classifier.save_model(model_path)
    return jsonify({'message': 'Model trained successfully'})

@app.route('/predict', methods=['POST'])
def predict():
    from predictor import Predictor
    from flask import request
    data = request.get_json()
    predictor = Predictor(model_path)
    prediction = predictor.predict(data['message'])
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
