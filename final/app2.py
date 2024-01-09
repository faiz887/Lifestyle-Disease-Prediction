# -*- coding: utf-8 -*-

import pickle
import numpy as np
from flask import Flask, request, render_template

# Load the pre-trained model
with open('model2.pkl', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('heart.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)
    output = prediction[0]
    if output == 1:
        return render_template('yes.html', prediction_text='The person is likely to have diabetes.')
    else:
        return render_template('heart.html', prediction_text='The person is not likely to have diabetes.')


if __name__ == '__main__':
        app.run(debug=True, port=5001)


