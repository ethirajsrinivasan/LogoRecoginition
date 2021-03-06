import tensorflow as tf
import common

PATCH_SIZE = 5
NUM_CLASSES = len(common.CLASS_NAME)


def params():
    # weights and biases
    params = {}

    params['w_conv1'] = tf.get_variable(
        'w_conv1',
        shape=[PATCH_SIZE, PATCH_SIZE, common.CNN_IN_CH, 32],
        initializer=tf.contrib.layers.xavier_initializer())
    params['b_conv1'] = tf.Variable(tf.constant(0.1, shape=[32]))

    params['w_conv2'] = tf.get_variable(
        'w_conv2',
        shape=[PATCH_SIZE, PATCH_SIZE, 32, 64],
        initializer=tf.contrib.layers.xavier_initializer())
    params['b_conv2'] = tf.Variable(tf.constant(0.1, shape=[64]))

    params['w_conv3'] = tf.get_variable(
        'w_conv3',
        shape=[PATCH_SIZE, PATCH_SIZE, 64, 128],
        initializer=tf.contrib.layers.xavier_initializer())
    params['b_conv3'] = tf.Variable(tf.constant(0.1, shape=[128]))

    params['w_fc1'] = tf.get_variable(
        'w_fc1',
        shape=[16 * 4 * 128, 2048],
        initializer=tf.contrib.layers.xavier_initializer())
    params['b_fc1'] = tf.Variable(tf.constant(0.1, shape=[2048]))

    params['w_fc2'] = tf.get_variable(
        'w_fc2',
        shape=[2048, NUM_CLASSES],
        initializer=tf.contrib.layers.xavier_initializer())
    params['b_fc2'] = tf.Variable(tf.constant(0.1, shape=[NUM_CLASSES]))

    return params


def cnn(data, model_params, keep_prob):
    # First layer
    h_conv1 = tf.nn.relu(
        tf.nn.conv2d(
            data, model_params['w_conv1'], [1, 1, 1, 1], padding='SAME') +
        model_params['b_conv1'])
    h_pool1 = tf.nn.max_pool(
        h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Second layer
    h_conv2 = tf.nn.relu(
        tf.nn.conv2d(
            h_pool1, model_params['w_conv2'], [1, 1, 1, 1], padding='SAME') +
        model_params['b_conv2'])
    h_pool2 = tf.nn.max_pool(
        h_conv2, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME')

    # Third layer
    h_conv3 = tf.nn.relu(
        tf.nn.conv2d(
            h_pool2, model_params['w_conv3'], [1, 1, 1, 1], padding='SAME') +
        model_params['b_conv3'])
    h_pool3 = tf.nn.max_pool(
        h_conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Fully connected layer
    conv_layer_flat = tf.reshape(h_pool3, [-1, 16 * 4 * 128])
    h_fc1 = tf.nn.relu(
        tf.matmul(conv_layer_flat, model_params['w_fc1']) +
        model_params['b_fc1'])
    h_fc1 = tf.nn.dropout(h_fc1, keep_prob)

    # Output layer
    out = tf.matmul(h_fc1, model_params['w_fc2']) + model_params['b_fc2']

    return out