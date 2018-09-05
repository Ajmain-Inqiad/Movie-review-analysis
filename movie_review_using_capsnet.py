# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 02:23:30 2018

@author: Shormi
"""

from keras import layers, models
from keras import backend as K
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
from keras.preprocessing import sequence
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.datasets import load_svmlight_files
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics import confusion_matrix

max_features = 5000
maxlen = 400
embed_dim = 50


def CapsNet(input_shape, n_class, num_routing):
    """
    A Capsule Network on MNIST.
    :param input_shape: data shape, 4d, [None, width, height, channels]
    :param n_class: number of classes
    :param num_routing: number of routing iterations
    :return: A Keras Model with 2 inputs and 2 outputs
    """
    x = layers.Input(shape=(maxlen,))
    embed = layers.Embedding(max_features, embed_dim, input_length=maxlen)(x)

    conv1 = layers.Conv1D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')(
        embed)

    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_vector]
    primarycaps = PrimaryCap(conv1, dim_vector=8, n_channels=32, kernel_size=9, strides=2, padding='valid')

    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_vector=16, num_routing=num_routing, name='digitcaps')(primarycaps)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    out_caps = Length(name='out_caps')(digitcaps)

    # Decoder network.
    y = layers.Input(shape=(n_class,))
    masked = Mask()([digitcaps, y])  # The true label is used to mask the output of capsule layer.
    x_recon = layers.Dense(512, activation='relu')(masked)
    x_recon = layers.Dense(1024, activation='relu')(x_recon)
    x_recon = layers.Dense(maxlen, activation='sigmoid')(x_recon)

    # two-input-two-output keras Model
    return models.Model([x, y], [out_caps, x_recon])


def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))


def train(model, data, args):
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param data: a tuple containing training and testing data, like `((x_train, y_train), (x_test, y_test))`
    :param args: arguments
    :return: The trained model
    """
    # unpacking the data
    (x_train, y_train), (x_test, y_test) = data

    # callbacks
    log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
                               batch_size=args.batch_size, histogram_freq=args.debug)
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5',
                                           save_best_only=True, save_weights_only=True, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: 0.001 * np.exp(-epoch / 10.))

    # compile the model
    model.compile(optimizer='adam',
                  loss=[margin_loss, 'mse'],
                  loss_weights=[1., args.lam_recon],
                  metrics={'out_caps': 'accuracy'})

    model.fit([x_train, y_train], [y_train, x_train], batch_size=args.batch_size, epochs=args.epochs,
              validation_data=[[x_test, y_test], [y_test, x_test]], callbacks=[log, tb, checkpoint], verbose=1)

    return model


def test(model, data):
    x_test, y_test = data
    y_pred, x_recon = model.predict([x_test, y_test], batch_size=100)
    print('-' * 50)
    print("Confusion Matrix for CapsNet")
    print(confusion_matrix(y_test, y_pred))
    
def load_files(files):
    return load_svmlight_files(files, n_features=None, dtype=None)
    
    
def tfidf(training_data, testing_data):
    tf_transformer = TfidfTransformer()
    #  It computes the TF for each review, the IDF using each review, and finally the TF-IDF for each review
    training_data_tfidf = tf_transformer.fit_transform(training_data)
    # .transform on the testing data which computes the TF for each review, 
    # then the TF-IDF for each review using the IDF from the training data 
    testing_data_tfidf = tf_transformer.transform(testing_data)

    return [training_data_tfidf,testing_data_tfidf]

    # Binerize target data

    # Converting target into binary
def binerize (raw_target):    
    binerize_target = []
    for i in range(len(raw_target)):
        if raw_target[i] > 5:
            binerize_target.append(1) # Positive
        else:
            binerize_target.append(0) # Negative
    return binerize_target    


def load_imdb(maxlen=400):
    files = ["dataset/train/labeledBow.feat","dataset/test/labeledBow.feat"]
    training_data, raw_training_target, testing_data, raw_testing_target = load_files(files)
    tfidf_data = tfidf(training_data, testing_data)

    training_data = tfidf_data[0]
    testing_data = tfidf_data[1]
    training_target = binerize(raw_training_target)
    testing_target = binerize(raw_testing_target)
    x_train = csr_matrix((training_data*100), dtype=np.int8).toarray()
    y_train = csr_matrix((training_target), dtype=np.int8).toarray()
    x_test = csr_matrix((testing_data*100), dtype=np.int8).toarray()
    y_test = csr_matrix((testing_target), dtype=np.int8).toarray()
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
    import os
    from keras import callbacks
    from keras.utils.vis_utils import plot_model

    # setting the hyper parameters
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--lam_recon', default=0.0005, type=float)
    parser.add_argument('--num_routing', default=3, type=int)  # num_routing should > 0
    parser.add_argument('--shift_fraction', default=0.1, type=float)
    parser.add_argument('--debug', default=0, type=int)  # debug>0 will save weights by TensorBoard
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('--is_training', default=1, type=int)
    parser.add_argument('--weights', default=None)
    args = parser.parse_args()
    print(args)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load data
    (x_train, y_train), (x_test, y_test) = load_imdb()
    
    y_train = y_train.ravel()
    y_test = y_test.ravel()
    # define model
    model = CapsNet(input_shape=x_train.shape,
                    n_class=1,
                    num_routing=args.num_routing)
    model.summary()
    train(model=model, data=((x_train, y_train), (x_test, y_test)), args=args)
    test(model=model, data=(x_test, y_test))
