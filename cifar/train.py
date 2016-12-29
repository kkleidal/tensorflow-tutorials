import tensorflow as tf
from utils import *
from nn import *
from model import * 
from data import *

LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1 # Initial learning rate.
NUM_EPOCHS_PER_DECAY = 50.0 # 350.0 # Epochs after which learning rate decays.
def optimizer(num_batches_per_epoch):
    global_step = tf.Variable(initial_value=0, trainable=False)
    increment_step = global_step.assign_add(1)
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True) 
    opt = tf.train.GradientDescentOptimizer(lr)
    return increment_step, opt, lr, global_step

def main(restore_if_possible=True, batch_size=128):
    with tf.device("/cpu:0"):
        # Build graph:
        image_batch, label_batch = input_graph(training=True, batch_size=batch_size)
        num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / batch_size
        increment_step, opt, lr, step = optimizer(num_batches_per_epoch)
        with tf.device("/cpu:0"): # Potentially gpu
            correct, loss = forward_propagation(image_batch, label_batch, train=True)
            grads = opt.compute_gradients(loss)
        with tf.control_dependencies([opt.apply_gradients(grads), increment_step]):
            train = tf.no_op(name='train')
        summaries = tf.summary.merge_all()

        # Train:
        saver = tf.train.Saver()  # For saving the model
        with tf.Session(config=tf.ConfigProto(
                log_device_placement=False)) as sess:
            summary_writer = tf.summary.FileWriter('tflog', sess.graph)  # For logging for TensorBoard

            # Initialize the variables (like the epoch counter).
            with tf.device("/cpu:0"): # Initialize variables on the main cpu
                sess.run(tf.global_variables_initializer())
            if restore_if_possible:
                try:
                    saver.restore(sess, tf.train.latest_checkpoint(SAVED_MODEL_DIR))
                except:
                    print("Couldn't find old model. Starting from scratch.")

            # Start input enqueue threads.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            try:
                while not coord.should_stop():
                    lrate, _, num_correct, batch_loss, i = sess.run([lr, train, correct, loss, step])
                    in_batch = i % num_batches_per_epoch
                    print("Epoch %d. Batch %d/%d. LR %.1e. Acc %.3f. Loss %.2f" % (i // num_batches_per_epoch, in_batch, num_batches_per_epoch, lrate, num_correct / float(batch_size), batch_loss))
                    if in_batch + 1 == num_batches_per_epoch: 
                        # Checkpoint, save the model:
                        summary = sess.run(summaries)
                        summary_writer.add_summary(summary)
                        saver.save(sess, SAVED_MODEL_PATH, global_step=i)

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
