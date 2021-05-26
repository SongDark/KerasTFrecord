import tensorflow as tf 
import numpy as np 

def make_example(image, label):
    features = {}
    features["image"] = tf.train.Feature(float_list=tf.train.FloatList(
        value=list(image.reshape(-1).astype(float) / 255.)))
    features["label"] = tf.train.Feature(float_list=tf.train.FloatList(
        value=[label]
    ))
    return tf.train.Example(features=tf.train.Features(feature=features))

def write_tfrecord(images, labels, target_file):
    assert len(images) == len(labels)
    writer = tf.io.TFRecordWriter(target_file)
    for i in range(len(images)):
        expample = make_example(images[i], labels[i]) 
        writer.write(expample.SerializeToString() )
    writer.close()

if __name__ == "__main__":
    mnist = np.load("./data/mnist.npz")
    write_tfrecord(mnist["x_train"], mnist["y_train"], "./data/mnist_train.tfrecords")
    write_tfrecord(mnist["x_test"], mnist["y_test"], "./data/mnist_test.tfrecords")