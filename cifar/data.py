import os
import tensorflow as tf

DIR = os.path.dirname(os.path.realpath(__file__))

IMAGE_SIZE = 24
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000
NUM_EXAMPLES_PER_EPOCH_FOR_CAT = 1

# Flip image/distort/etc
def _image_processing(image, train=True, brightness_distortion=63, contrast_distortion_lower=0.2, contrast_distortion_upper=1.8):
    # Cropping
    if train:
        image = tf.random_crop(image, [IMAGE_SIZE, IMAGE_SIZE, 3])
        # Flipping
        image = tf.image.random_flip_left_right(image)
        # Brightness distortion
        image = tf.image.random_brightness(image, brightness_distortion)
        # Brightness distortion
        image = tf.image.random_contrast(image, contrast_distortion_lower, contrast_distortion_upper)
    else:
        image = tf.image.resize_image_with_crop_or_pad(image, IMAGE_SIZE, IMAGE_SIZE)
    # Whitening
    image = tf.image.per_image_standardization(image)
    image.set_shape([IMAGE_SIZE, IMAGE_SIZE, 3])
    return image

# Graph ops for loading, parsing, and queuing training images 
def input_graph(training=True, partition='test', batch_size=100):
    with tf.name_scope("input"):
        if training or partition == 'train':
            # The training image files:
            filenames = [os.path.join(DIR, "cifar-10-batches-bin", "data_batch_%d.bin" % i) for i in range(1,6)]
            num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
        elif partition == 'test':
            filenames = [os.path.join(DIR, "cifar-10-batches-bin", 'test_batch.bin')]
            num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
        elif partition == 'cat':
            filenames = [os.path.join(DIR, "cat", 'cat-small.bin')]
            num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_CAT
        else:
            assert(False)
        filename_queue = tf.train.string_input_producer(filenames)
        # Read each 3073 byte record:
        reader = tf.FixedLengthRecordReader(3073)
        key, record_string = reader.read(filename_queue)
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
            image = (tf.cast(tf.transpose(record_input, [1, 2, 0]), tf.float32) - 128.0) / 256.0
            image = _image_processing(image, train=training)
        with tf.name_scope("batching"):
            # Load 10000 images to start, then continue enqueuing up to capacity
            min_after_dequeue = int(num_examples_per_epoch * 0.4)
            capacity = min_after_dequeue + 3 * batch_size
            kwargs = dict(batch_size=batch_size, capacity=capacity)
            if training:
                batch_fn = tf.train.shuffle_batch
                kwargs["min_after_dequeue"] = min_after_dequeue
            else:
                batch_fn = tf.train.batch
            image_batch, label_batch = batch_fn(
                [image, record_label], **kwargs)

            # The examples and labels for training a single batch
            tf.summary.image("image", image_batch, max_outputs=3)
            labels = tf.squeeze(label_batch, axis=1)
            return image_batch, labels, num_examples_per_epoch
