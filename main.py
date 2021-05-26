# encoding:utf-8
import tensorflow as tf 
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from networks import build_mlp
from tfrecord_loader import load_dataset

if __name__ == "__main__":

    batch_size = 100
    epochs = 50
    save_path = "./models/model.h5"

    train_dataset = load_dataset("./data/mnist_train.tfrecords", 1, batch_size)
    val_dataset = load_dataset("./data/mnist_test.tfrecords", 1, batch_size)

    earlystop = EarlyStopping("val_accuracy", mode="max", patience=3, verbose=1)
    checkpoint = ModelCheckpoint(save_path, "val_accuracy", mode="max", save_best_only=True, verbose=1, save_weights_only=True)

    model = build_mlp(input_shape=(28 * 28,), num_classes=10)
    model.compile("adam", "sparse_categorical_crossentropy", ["sparse_categorical_crossentropy", "accuracy"])
    model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, callbacks=[earlystop, checkpoint])
    