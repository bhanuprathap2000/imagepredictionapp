# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 20:15:38 2020

@author: Bhanu
"""

from tensorflow.keras.applications.vgg19 import VGG19
model=VGG19(weights="imagenet")

model.save("VGG19.h5")