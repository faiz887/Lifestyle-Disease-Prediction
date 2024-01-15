import numpy as np
from flask import Flask, render_template, request
from keras.preprocessing.image import load_img, img_to_array
import os
from keras.models import load_model

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def read_image(filename):
    img = load_img(filename, color_mode='rgb', target_size=(64, 64))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img


@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template('home.html')


@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = file.filename
            file_path = os.path.join('static/images', filename)
            file.save(file_path)
            img = read_image(file_path)

            # Load the Keras model for prediction
            model = load_model("BrainTumor10_epochs.h5py")

            # Predict the class of an image
            predictions = model.predict(img)
            class_prediction = np.argmax(predictions, axis=-1)

            # Map predicted class to human-readable format
            if class_prediction[0] == 0:
                product = "no pneumonia"
            elif class_prediction[0] == 1:
                product = "pneumonia"

            return render_template('predict.html', product=product, user_image=file_path)

    return render_template('predict.html')


if __name__ == "__main__":
    app.run(debug=True,port=5003)
