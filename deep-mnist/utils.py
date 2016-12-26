import tensorflow as tf

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        yield tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        yield tf.summary.scalar('stddev', stddev)
        yield tf.summary.scalar('max', tf.reduce_max(var))
        yield tf.summary.scalar('min', tf.reduce_min(var))
        yield tf.summary.histogram('histogram', var)

def weight_variable(dims):
    with tf.device("/cpu:0"):
        return tf.Variable(tf.truncated_normal(dims, stddev=0.1), name="weights")

def bias_variable(dim):
    with tf.device("/cpu:0"):
        return tf.Variable(tf.constant(0.1, shape=[dim]), name="bias")
