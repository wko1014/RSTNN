# Import APIs
import tensorflow as tf

# Tensorflow 1.5-based

def RSTNN(eeg, label, num_channel, num_output, reuse=False, seizure_experiment=False):
    with tf.variable_scope("RSTNN", reuse=reuse):
        if reuse == True: keep_prob = 1.0
        else: keep_prob = 0.5

        hidden1 = tf.layers.conv2d(inputs=eeg, filters=16, kernel_size=(1, 9), padding="same", activation=tf.nn.relu)
        hidden1 = tf.layers.batch_normalization(hidden1)
        hidden2 = tf.layers.conv2d(inputs=hidden1, filters=16, kernel_size=(1, 9), padding="same", activation=tf.nn.relu)
        hidden = hidden1 + hidden2
        hidden = tf.layers.batch_normalization(hidden)
        hidden3 = tf.layers.conv2d(inputs=hidden, filters=16, kernel_size=(1, 9), padding="same", activation=tf.nn.relu)
        hidden3 = tf.layers.batch_normalization(hidden3)
        hidden3 = tf.layers.max_pooling2d(hidden3, (1, 4), (1, 4))
        hidden3 = tf.layers.dropout(hidden3, rate=keep_prob)

        hidden4 = tf.layers.conv2d(inputs=hidden3, filters=32, kernel_size=(1, 1), padding="same", activation=tf.nn.relu)
        hidden4 = tf.layers.batch_normalization(hidden4)
        hidden5 = tf.layers.conv2d(inputs=hidden4, filters=32, kernel_size=(1, 9), padding="same", activation=tf.nn.relu)
        hidden = hidden4 + hidden5
        hidden = tf.layers.batch_normalization(hidden)
        hidden6 = tf.layers.conv2d(inputs=hidden, filters=32, kernel_size=(1, 9), padding="same", activation=tf.nn.relu)
        hidden += hidden6
        hidden = tf.layers.batch_normalization(hidden)
        hidden7 = tf.layers.conv2d(inputs=hidden, filters=32, kernel_size=(1, 9), padding="same", activation=tf.nn.relu)
        hidden7 = tf.layers.batch_normalization(hidden7)
        hidden7 = tf.layers.max_pooling2d(hidden7, (1, 4), (1, 4))
        hidden7 = tf.layers.dropout(hidden7, rate=keep_prob)

        hidden8 = tf.layers.conv2d(inputs=hidden7, filters=64, kernel_size=(1, 1), padding="same", activation=tf.nn.relu)
        hidden8 = tf.layers.batch_normalization(hidden8)
        hidden9 = tf.layers.conv2d(inputs=hidden8, filters=64, kernel_size=(1, 9), padding="same", activation=tf.nn.relu)
        hidden = hidden8 + hidden9
        hidden = tf.layers.batch_normalization(hidden)
        hidden10 = tf.layers.conv2d(inputs=hidden, filters=64, kernel_size=(1, 9), padding="same", activation=tf.nn.relu)
        hidden += hidden10
        hidden = tf.layers.batch_normalization(hidden)
        feature = tf.layers.conv2d(inputs=hidden, filters=64, kernel_size=(1, 9), padding="same", activation=tf.nn.relu)
        feature = tf.layers.batch_normalization(feature)
        feature = tf.layers.max_pooling2d(feature, (1, 4), (1, 4))
        feature = tf.layers.dropout(feature, rate=keep_prob)

        feature = tf.layers.conv2d(inputs=feature, filters=64, kernel_size=(num_channel, 1), padding="valid", activation=tf.nn.leaky_relu)
        feature = tf.layers.batch_normalization(feature)

        feature = tf.layers.flatten(feature)
        feature = tf.layers.dense(feature, 512, activation=tf.nn.relu)
        feature = tf.layers.dropout(feature, rate=keep_prob)

        feature = tf.layers.dense(feature, 64, activation=tf.nn.relu)
        feature = tf.layers.dropout(feature, rate=keep_prob)

        feature = tf.layers.dense(feature, num_output, activation=None)
        if num_output == 1:
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=feature))
            prediction = tf.math.round(tf.math.sigmoid(feature))
            probability = tf.math.sigmoid(feature)
            if seizure_experiment == True:
                return loss, prediction, probability
        else:
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=feature))
            prediction = tf.math.round(tf.math.softmax(feature))
    return loss, prediction
