# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 20:06:48 2020

@author: Bhanu
"""

import os
import numpy as np
from tensorflow import keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input,decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from flask import Flask,redirect,url_for,request,render_template
from tensorflow.keras.applications.vgg19 import VGG19
model=VGG19(weights="imagenet")

model.save("VGG19.h5")
model=VGG19(weights="imagenet")
app=Flask(__name__)



model=load_model(model)


def model_predict(img_path,model):
    img=image.load_img(img_path,target_size=(224,224))
    x=np.expand_dims(img, axis=0)
    x=preprocess_input(x)
    preds=model.predict(x)
    return preds
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")
@app.route("/predict",methods=["GET","POST"])
def predict():
    if request.method=="POST":
        f=request.files["file"]
        basepath=os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', f.filename)
        f.save(file_path)
        preds=model_predict(file_path,model)
        pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        result = str(pred_class[0][0][1])               # Convert to string
        return render_template("predict.html",imageprediction=result)


if __name__=="__main__":
    app.run()
