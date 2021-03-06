import tensorflow as tf
from utils import *
from nn import *
from model import * 
from data import *

def evaluate(partition="train", batch_size=100):
    g = tf.Graph()
    with g.as_default():
        with tf.device("/cpu:0"):
            # Build graph:
            image_batch, label_batch, num_examples_per_epoch = input_graph(training=False, partition=partition, batch_size=batch_size)
            img = tf.summary.image("image", image_batch, 1)
            with tf.device("/cpu:0"): # Potentially gpu
                correct, loss, proba = forward_propagation(image_batch, label_batch)
            proba_summary = tf.summary.tensor_summary("proba", proba)
            summaries = tf.summary.merge([img, proba_summary])

            restorer = tf.train.Saver()  # For saving the model
            acc = 0.0
            with tf.Session(config=tf.ConfigProto(
                    log_device_placement=False)) as sess:
                # Initialize the variables (like the epoch counter).
                with tf.device("/cpu:0"): # Initialize variables on the main cpu
                    sess.run(tf.global_variables_initializer())
                restorer.restore(sess, tf.train.latest_checkpoint(SAVED_MODEL_DIR))

                if partition == "cat":
                    summary_writer = tf.summary.FileWriter('tflog-eval', sess.graph)  # For logging for TensorBoard

                # Start input enqueue threads.
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)

                try:
                    i = 0
                    num_correct = 0
                    while i * batch_size < num_examples_per_epoch \
                            and not coord.should_stop():
                        if partition == "cat":
                            current_correct, summary, pr = sess.run([correct, summaries, proba])
                            classes = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
                            for cls, prb in zip(classes, pr[0,:].tolist()):
                                print "%s: %6.2f%%" % (cls, prb * 100.0)
                            summary_writer.add_summary(summary)
                        else:
                            current_correct = sess.run(correct)
                        num_correct += current_correct
                        i += 1
                    total = i * batch_size
                    acc = num_correct / float(total)
                except tf.errors.OutOfRangeError:
                    print('Done training -- epoch limit reached')
                finally:
                    # When done, ask the threads to stop.
                    coord.request_stop()

                # Wait for threads to finish.
                coord.join(threads)
                sess.close()
            return acc

def main(batch_size=100):
    test_acc = evaluate(partition="cat")
    print("Cat accuracy: %.3f" % (100.0 * test_acc))
    train_acc = evaluate(partition="train")
    print("Training accuracy: %.3f" % (100.0 * train_acc))
    test_acc = evaluate(partition="test")
    print("Test accuracy: %.3f" % (100.0 * test_acc))

if __name__ == "__main__":
    main()
