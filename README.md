# SMS Spam Detection Service

This repository contains a Flask-based web service for SMS spam detection. The service utilizes a Random Forest Classifier and is designed to predict whether an SMS text message is "spam" or "ham" (not spam).

## Project Structure

The project contains the following main files:

1. `model.py` - This file defines the `RandomForestSpamClassifier` class, which encapsulates all the logic related to training the spam detection model and making predictions.

2. `predictor.py` - This file defines the `Predictor` class, which handles loading the trained model and making predictions.

3. `app.py` - This is the main Flask application file. It defines the `/train` and `/predict` endpoints.

## Setup

To set up and run the project, follow these steps:

1. Clone the repository and navigate to the project directory.

2. Create a virtual environment and activate it:

    ```
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install the required Python packages:

    ```
    pip install -r requirements.txt
    ```

4. Run the Flask application:

    ```
    python app.py
    ```

## Usage

The service exposes two main endpoints:

### `/train` (GET)

This endpoint trains the model using the data from `spam_randomForest.csv` and saves the trained model. There is no need to provide any data to this endpoint.

Example:
curl -X GET http://localhost:5001/train


### `/predict` (POST)

This endpoint makes a prediction based on the message provided in the POST request.

Example:
curl -X POST -H "Content-Type: application/json" -d '{"message": "Hey, are we still on for dinner tonight?"}' http://localhost:5001/predict


The service will respond with a prediction of either "ham" or "spam".

## Contributing

If you would like to contribute to this project, feel free to submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)

