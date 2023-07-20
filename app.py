from flask import Flask, request, jsonify
from model import RandomForestSpamClassifier
from predictor import Predictor
import pandas as pd

app = Flask(__name__)
model_path = 'models/model.pkl'
classifier = RandomForestSpamClassifier()

'''
@app.route('/train', methods=['GET'])
def train():
    df = pd.read_csv('datasets/spam_randomForest.csv', encoding='latin-1')
    df = df[['v2', 'v1']]
    df = df.rename(columns={'v2': 'messages', 'v1': 'label'})
    classifier.train(df)
    classifier.save_model(model_path)
    return jsonify({'message': 'Model trained successfully'})
'''
@app.route('/train', methods=['GET'])
def train():
    # Load the first old dataset
    df_old = pd.read_csv('datasets/spam_randomForest.csv', encoding='latin-1')
    df_old = df_old[['v2', 'v1']]
    df_old = df_old.rename(columns={'v2': 'messages', 'v1': 'label'})

    # Load the second old dataset
    df_new = pd.read_csv('datasets/SMSSmishCollection.txt', sep='\t', header=None, names=['label', 'messages'])

    # Replace 'smish' with 'spam' in the label column
    df_new['label'] = df_new['label'].replace('smish', 'spam')

    # Load the new dataset
    df_newest = pd.read_csv('datasets/Dataset_5971.csv', encoding='latin-1')
    df_newest = df_newest[['TEXT', 'LABEL']]
    df_newest = df_newest.rename(columns={'TEXT': 'messages', 'LABEL': 'label'})

    # Replace 'Smishing' with 'spam' in the label column
    df_newest['label'] = df_newest['label'].replace('Smishing', 'spam')

    # Combine all the datasets
    df = pd.concat([df_old, df_new, df_newest])

    # Train the model and save it
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
