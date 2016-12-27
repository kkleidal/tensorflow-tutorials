import os
import tensorflow as tf
from utils import *

DIR = os.path.dirname(os.path.realpath(__file__))
SAVED_MODEL_PATH = os.path.join(DIR, "model", "model.ckpt")

# Flip image/distort/etc
def processing(image):
    return image

# Graph ops for loading, parsing, and queuing training images 
def training_input_graph(batch_size=100):
    with tf.name_scope("input"):
        # The training image files:
        train_filename_queue = tf.train.string_input_producer([os.path.join(DIR, "cifar-10-batches-bin", "data_batch_%d.bin" % i) for i in range(1,6)])
        # Read each 3073 byte record:
        reader = tf.FixedLengthRecordReader(3073)
        key, record_string = reader.read(train_filename_queue)
        with tf.name_scope("parsing"):
            # Convert the string to a byte array, the first byte is the class label,
            # the next 3072 bytes are a 32x32 image with 3 channels:
            #  Axes: (channel, row, col)
            record_bytes = tf.decode_raw(record_string, tf.uint8)
            record_label = tf.cast(tf.slice(record_bytes, [0], [1]), tf.int64)
            record_input = tf.reshape(
                          tf.slice(record_bytes, [1], [3072]),
                          [3, 32, 32])
        with tf.name_scope("processing"):
            # flipping/etc
            image = processing(record_input)
        with tf.name_scope("batching"):
            # Load 10000 images to start, then continue enqueuing up to capacity
            min_after_dequeue = 10000
            capacity = min_after_dequeue + 3 * batch_size
            image_batch, label_batch = tf.train.shuffle_batch(
                [image, record_label], batch_size=batch_size, capacity=capacity,
                min_after_dequeue=min_after_dequeue)
            # The examples and labels for training a single batch
            return image_batch, label_batch

def train_graph(image_batch, label_batch):
    labels = tf.squeeze(label_batch, axis=1)
    input_layer = tf.cast(tf.reshape(image_batch, [tf.shape(image_batch)[0], 3 * 32 * 32]), tf.float32)

    with tf.name_scope('class-output'):
        with tf.name_scope('weights'):
            weights = weight_variable([3 * 32 * 32, 10])
        with tf.name_scope('biases'):
            bias = bias_variable(10)
        with tf.name_scope('activation'):
            activation = tf.matmul(input_layer, weights) + bias
            logits = activation
        with tf.name_scope('output'):
            with tf.name_scope('probabilities'):
                proba = tf.nn.softmax(logits)
            with tf.name_scope('predictions'):
                prediction = tf.argmax(proba, 1)

    with tf.name_scope('accuracy'):
        with tf.name_scope('accuracy'):
            actual = labels
            with tf.name_scope('num_correct'):
                correct = tf.reduce_sum(tf.to_int32(tf.equal(prediction, actual)))

    with tf.name_scope('loss'):
        labels_one_hot = tf.one_hot(labels, 10, on_value=1.0, off_value=0.0)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels_one_hot))

    train = tf.train.AdamOptimizer(0.01).minimize(loss, name="train")
    return train, correct, loss 

def main():
    image_batch, label_batch = training_input_graph()
    train, correct, loss = train_graph(image_batch, label_batch)
    saver = tf.train.Saver()  # For saving the model
    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter('tflog', sess.graph)  # For logging for TensorBoard

        # Initialize the variables (like the epoch counter).
        with tf.device("/cpu:0"): # Initialize variables on the main cpu
            sess.run(tf.global_variables_initializer())

        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            i = 0
            while not coord.should_stop():
                _, num_correct, batch_loss = sess.run([train, correct, loss])
                print "Iteration %d. Acc %.3f. Loss %.2f" % (i, num_correct / 100.0, batch_loss)
                i += 1
                # Checkpoint, save the model:
                saver.save(sess, SAVED_MODEL_PATH, global_step=(i+1))

        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()

        # Wait for threads to finish.
        coord.join(threads)
        sess.close()

main()