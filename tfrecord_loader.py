
import tensorflow as tf 

def input_fn(example):
    proto = {
        "image": tf.io.FixedLenFeature([28*28, ], tf.float32),
        "label": tf.io.FixedLenFeature([], tf.float32),
    }
    example = tf.io.parse_single_example(example, proto)
    image = example["image"]
    label = example["label"]
    assert image is not None 
    return image, label 

def load_dataset(files, epochs, batchsize):

    dataset = tf.data.TFRecordDataset(
        files
    )
    dataset = dataset.map(input_fn)
    dataset = dataset.shuffle(buffer_size=5 * batchsize, reshuffle_each_iteration=True)
    dataset = dataset.repeat(epochs)
    dataset = dataset.batch(batchsize)
    dataset = dataset.prefetch(buffer_size=5 * batchsize)
    return dataset