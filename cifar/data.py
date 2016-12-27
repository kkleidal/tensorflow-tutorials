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

def conv1_layer(input_to_layer, name='conv1-layer'):
    with tf.variable_scope(name, values=[input_to_layer]):
        with tf.name_scope('weights'):
            weights = weight_variable([5, 5, 3, 64], stddev=5e-2)
        with tf.name_scope('biases'):
            bias = bias_variable(64)
        with tf.name_scope('preactivation'):
            conv = tf.nn.conv2d(input_to_layer, weights, [1, 1, 1, 1], padding='SAME')
            preactivation = tf.nn.bias_add(conv, bias)
        with tf.name_scope('output'):
            out = tf.nn.relu(preactivation)
    return out

def softmax_layer(input_to_layer, name='softmax-layer'):
    with tf.variable_scope(name, values=[input_to_layer]):
        with tf.name_scope('weights'):
            weights = weight_variable([input_to_layer.get_shape()[1], 10])
        with tf.name_scope('biases'):
            bias = bias_variable(10)
        with tf.name_scope('preactivation'):
            preactivation = tf.matmul(input_to_layer, weights) + bias
            logits = preactivation
        with tf.name_scope('output'):
            with tf.name_scope('probabilities'):
                proba = tf.nn.softmax(logits)
            with tf.name_scope('predictions'):
                prediction = tf.argmax(logits, 1)
    return logits, proba, prediction

def forward_propagation(image_batch, label_batch):
    images = tf.transpose(tf.cast(image_batch, tf.float32), [0, 2, 3, 1])
    tf.summary.image("image", images, max_outputs=3)
    labels = tf.squeeze(label_batch, axis=1)

    conv1 = conv1_layer(images)

    flattened = tf.reshape(conv1, [tf.shape(conv1)[0], 32 * 32 * 64])

    logits, proba, prediction = softmax_layer(flattened)

    with tf.name_scope('accuracy'):
        with tf.name_scope('accuracy'):
            actual = labels
            with tf.name_scope('num_correct'):
                correct = tf.reduce_sum(tf.to_int32(tf.equal(prediction, actual)))

    with tf.name_scope('loss'):
        labels_one_hot = tf.one_hot(labels, 10, on_value=1.0, off_value=0.0)
        loss = tf.nn.softmax_cross_entropy_with_logits(logits, labels_one_hot)
        batch_loss = tf.reduce_mean(loss)

    return correct, batch_loss

def main(batch_size=2):
    with tf.device("/cpu:0"):
        # Build graph:
        opt = tf.train.GradientDescentOptimizer(0.01)
        image_batch, label_batch = training_input_graph(batch_size=batch_size)
        with tf.device("/cpu:0"):
            correct, loss = forward_propagation(image_batch, label_batch)
            grads = opt.compute_gradients(loss)
        train = opt.apply_gradients(grads)
        summaries = tf.summary.merge_all()

        # Train:
        saver = tf.train.Saver()  # For saving the model
        with tf.Session(config=tf.ConfigProto(
                log_device_placement=True)) as sess:
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
                    _, num_correct, batch_loss, summary = sess.run([train, correct, loss, summaries])
                    print("Iteration %d. Acc %.3f. Loss %.2f" % (i, num_correct / batch_size, batch_loss))
                    i += 1
                    # Checkpoint, save the model:
                    saver.save(sess, SAVED_MODEL_PATH, global_step=(i+1))
                    summary_writer.add_summary(summary)

            except tf.errors.OutOfRangeError:
                print('Done training -- epoch limit reached')
            finally:
                # When done, ask the threads to stop.
                coord.request_stop()

            # Wait for threads to finish.
            coord.join(threads)
            sess.close()

main()
