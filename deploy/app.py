from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import cv2

# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf

from tensorflow.compat.v1.keras.backend import set_session

config = tf.compat.v1.ConfigProto()

config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU

config.log_device_placement = True  # to log device placement (on which device the operation ran)

sess = tf.compat.v1.Session(config=config)

set_session(sess)

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH ='models/vgg19.h5'

# Load your trained model
import keras
model = load_model(MODEL_PATH,custom_objects={'LeakyReLU':keras.layers.LeakyReLU}) #Change to your defined activation function while building the model
                                                                                   


def model_predict(img_path, model):
    IMG_SIZE = 224
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    

    # Preprocessing the image
    x1 = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    # x = np.true_divide(x, 255)
    ## Scaling

   

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = x1.reshape(-1, IMG_SIZE, IMG_SIZE,3)

    preds = model.predict(x)
    preds= np.argmax(preds,axis=-1)
    if preds[0]==0:
        preds="Basket"
    elif preds[0]==1:
        preds="Coin"
    else:
        preds="Figure"
    
    
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['image']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result=preds
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)