import os
import numpy as np
import tensorflow as tf
import tqdm
from tensorflow.examples.tutorials.mnist import input_data
from utils import *
from model import *

def main(alpha=0.9):
    graph = build_graph(GraphConfig()) 
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    with tf.Session() as sess:
        # Load the model:
        restorer = tf.train.Saver(tf.global_variables()) # tf.train.import_meta_graph(SAVED_MODEL_PATH + ".meta")
        restorer.restore(sess, tf.train.latest_checkpoint(SAVED_MODEL_DIR))
        # Test error:
        test_acc = accuracy(sess, graph, mnist.test.images, mnist.test.labels)
        print("Test accuracy: %.4f" % test_acc)

if __name__ == "__main__":
    main()
