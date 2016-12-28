import tensorflow as tf
from utils import *
from nn import *
from model import * 
from data import *

def main(batch_size=100):
    with tf.device("/cpu:0"):
        # Build graph:
        opt = tf.train.AdamOptimizer(0.01)
        image_batch, label_batch = input_graph(training=True, batch_size=batch_size)
        with tf.device("/cpu:0"): # Potentially gpu
            correct, loss = forward_propagation(image_batch, label_batch, train=True)
            grads = opt.compute_gradients(loss)
        train = opt.apply_gradients(grads)
        summaries = tf.summary.merge_all()

        # Train:
        saver = tf.train.Saver()  # For saving the model
        with tf.Session(config=tf.ConfigProto(
                log_device_placement=False)) as sess:
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
                    print("Iteration %d. Acc %.3f. Loss %.2f" % (i, num_correct / float(batch_size), batch_loss))
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

if __name__ == "__main__":
    main()
