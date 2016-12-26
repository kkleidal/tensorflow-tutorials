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
    total = 1e6
    batch_size = 200
    iterations = int(total // batch_size)
    saver = tf.train.Saver()  # For saving the model
    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter('tflog', sess.graph)  # For logging for TensorBoard
        with tf.device("/cpu:0"): # Initialize variables on the main cpu
            sess.run(tf.global_variables_initializer())
        for i in tqdm.tqdm(range(iterations)):
            inputs, labels = mnist.train.next_batch(batch_size)
            if i % 100 == 0:
                # Checkpoint, save the model:
                saver.save(sess, SAVED_MODEL_PATH, global_step=(i+1))
                # Train error:
                train_acc = accuracy(sess, graph, inputs, labels)
                print("Batch %6d train accuracy: %.4f" % (i, train_acc))
            # Train on the batch:
            sess.run(graph.train, feed_dict={
                graph.inputs: inputs,
                graph.labels: labels,
                graph.dropout_keep_prob: 0.8,
            })
        # Test error:
        test_acc = accuracy(sess, graph, mnist.test.images, mnist.test.labels)
        print("Test accuracy: %.4f" % test_acc)

if __name__ == "__main__":
    main()
