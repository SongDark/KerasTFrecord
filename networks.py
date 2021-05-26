# encoding:utf-8
import tensorflow as tf 
from tensorflow.python.keras.layers import Flatten, Dense, Dropout
from tensorflow.python.keras.models import Sequential

def build_mlp(input_shape, num_classes):
    layers = [
        Flatten(input_shape=input_shape),

        Dense(units=128, activation="relu"),
        Dropout(0.5), 

        Dense(units=64, activation="relu"),
        Dropout(0.5), 

        Dense(units=32, activation="relu"),
        Dropout(0.5), 

        Dense(units=num_classes, activation="softmax")
    ]
    return Sequential(layers)