# Train Keras Model using Tfrecords

## Prepare Data

Get `mnist.npz` from here:

```shell
    wget https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
```

Then run the following command to generate `mnist_train.tfrecords` and `mnist_test.tfrecords`.

```shell
    python dataprocess.py
```

## Train With Tfrecords

```shell
    main.py
```