import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from utils import *

class GraphConfig:
    learning_rate = 0.002
    conv1_size = 32
    conv2_size = 64
    fc_size = 1024

def ReLuLayer(inp, inp_size, size, summaries=None):
    if summaries is None:
        summaries = []
    with tf.name_scope('weights'):
        weights = weight_variable([inp_size, size])
        summaries.extend(variable_summaries(weights))
    with tf.name_scope('biases'):
        bias = bias_variable(size)
        summaries.extend(variable_summaries(bias))
    with tf.name_scope('activation'):
        activation = tf.matmul(inp, weights) + bias
        summaries.append(tf.summary.histogram("activation", activation))
    with tf.name_scope('output'):
        hidden = tf.nn.relu(activation)
        summaries.append(tf.summary.histogram("output", hidden))
    return hidden

def ConvLayer(inp, inp_channels=1, size=32, summaries=None):
    if summaries is None:
        summaries = []
    with tf.name_scope('weights'):
        W = weight_variable([5, 5, inp_channels, size])
        summaries.extend(variable_summaries(W))
    with tf.name_scope('biases'):
        bias = bias_variable(size)
        summaries.extend(variable_summaries(bias))
    with tf.name_scope('activation'):
        conv = tf.nn.conv2d(inp, W, strides=[1, 1, 1, 1], padding='SAME')
        activation = tf.nn.relu(conv + bias)
        summaries.append(tf.summary.histogram("activation", activation))
    with tf.name_scope('output'):
        hidden = tf.nn.relu(activation)
        summaries.append(tf.summary.histogram("output", hidden))
    return hidden

def MaxPoolLayer(inp, summaries=None):
    return tf.nn.max_pool(inp, ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1], padding='SAME')

def build_graph(cfg):
    tf.set_random_seed(42)

    labels = tf.placeholder(tf.float32, [None, 10], name="labels")

    train_summaries = []

    with tf.name_scope('input'):
        inputs = tf.placeholder(tf.float32, [None, 784], name="inputs")
        images = tf.reshape(inputs, [tf.shape(inputs)[0], 28, 28, 1])

    with tf.name_scope('conv1'):
        conv1 = ConvLayer(images, inp_channels=1, size=cfg.conv1_size, summaries=train_summaries) 

    with tf.name_scope('max-pool1'):
        max1 = MaxPoolLayer(conv1, summaries=train_summaries)

    with tf.name_scope('conv2'):
        conv2 = ConvLayer(max1, inp_channels=cfg.conv1_size, size=cfg.conv2_size, summaries=train_summaries) 

    with tf.name_scope('flattening'):
        flat_size = 14 * 14 * cfg.conv2_size
        flat = tf.reshape(conv2, [tf.shape(conv2)[0], flat_size])

    with tf.name_scope('fully-connected'):
        hidden_full = ReLuLayer(flat, flat_size, cfg.fc_size, summaries=train_summaries)
        keep_prob = tf.placeholder(tf.float32, name="dropout-keep-prob")
        hidden = tf.nn.dropout(hidden_full, keep_prob)

    with tf.name_scope('class-output'):
        with tf.name_scope('weights'):
            weights = weight_variable([cfg.fc_size, 10])
            train_summaries.extend(variable_summaries(weights))
        with tf.name_scope('biases'):
            bias = bias_variable(10)
            train_summaries.extend(variable_summaries(bias))
        with tf.name_scope('activation'):
            activation = tf.matmul(hidden, weights) + bias
            train_summaries.append(tf.summary.histogram("activation", activation))
            logits = activation
        with tf.name_scope('output'):
            with tf.name_scope('probabilities'):
                proba = tf.nn.softmax(logits)
            with tf.name_scope('predictions'):
                prediction = tf.argmax(proba, 1)
            train_summaries.append(tf.summary.histogram("proba", proba))
            train_summaries.append(tf.summary.histogram("prediction", prediction))

    with tf.name_scope('accuracy'):
        with tf.name_scope('accuracy'):
            actual = tf.argmax(labels, 1, name="actual")
            with tf.name_scope('num_correct'):
                correct = tf.reduce_sum(tf.to_int32(tf.equal(prediction, actual)))
                train_summaries.append(tf.summary.scalar("num_correct", correct))

    with tf.name_scope('loss'):
        class_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels))
        train_summaries.append(tf.summary.scalar("loss", class_loss))
        total_loss = class_loss
        

    train = tf.train.AdamOptimizer(cfg.learning_rate).minimize(total_loss, name="train")

    g = lambda: None
    g.cfg = cfg
    g.inputs = inputs
    g.labels = labels
    g.proba = proba
    g.prediction = prediction
    g.loss = class_loss
    g.train = train
    g.correct = correct
    g.summaries = tf.summary.merge(train_summaries)
    g.dropout_keep_prob = keep_prob

    return g

def accuracy(sess, graph, images, labels):
    batch_size = 1000
    i = 0
    total_correct = 0
    while i * batch_size <= len(labels):
        total_correct += sess.run(graph.correct, feed_dict={
            graph.inputs: images[i*batch_size : (i+1)*batch_size],
            graph.labels: labels[i*batch_size : (i+1)*batch_size],
            graph.dropout_keep_prob: 1.0,
        })
        i += 1
    return total_correct / float(len(labels))


def main(alpha=0.9):
    graph = build_graph(GraphConfig()) 
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    batch_size = 50
    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter('tflog', sess.graph)
        sess.run(tf.global_variables_initializer())
        for i in xrange(20000):
            inputs, labels = mnist.train.next_batch(batch_size)
            if i % 100 == 0:
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
