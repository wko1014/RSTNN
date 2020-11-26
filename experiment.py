# Import APIs
import tensorflow as tf
import numpy as np
import utils
import network

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

def experiment(dataset_name, network_name, subject, session_fold, print_detail=False, num_epochs=100, batch_size=16, learning_rate=1e-3):
    # Placeholding
    if dataset_name == "GIST-MI":
        num_output = 1
        num_channel = 64
        sfreq = 512

        X_train = tf.placeholder(tf.float32, [None, num_channel, 1024, 1])
        Y_train = tf.placeholder(tf.float32, [None, num_output])

        X_valid = tf.placeholder(tf.float32, [None, num_channel, 1024, 1])
        Y_valid = tf.placeholder(tf.float32, [None, num_output])

        X_test = tf.placeholder(tf.float32, [None, num_channel, 1024, 1])
        Y_test = tf.placeholder(tf.float32, [None, num_output])
    elif dataset_name == "KU-MI":
        num_output = 2
        num_channel = 62
        sfreq = 1000

        X_train = tf.placeholder(tf.float32, [None, num_channel, 3000, 1])
        Y_train = tf.placeholder(tf.int32, [None, num_output])

        X_valid = tf.placeholder(tf.float32, [None, num_channel, 3000, 1])
        Y_valid = tf.placeholder(tf.int32, [None, num_output])

        X_test = tf.placeholder(tf.float32, [None, num_channel, 3000, 1])
        Y_test = tf.placeholder(tf.int32, [None, num_output])

    # Call neural network
    if network_name == "RSTNN":
        training_loss, train_prediction = network.RSTNN(eeg=X_train, label=Y_train, num_channel=num_channel, num_output=num_output)
        _, valid_prediction = network.RSTNN(eeg=X_valid, label=Y_valid, num_channel=num_channel, num_output=num_output, reuse=True)
        _, test_prediction = network.RSTNN(eeg=X_test, label=Y_test, num_channel=num_channel, num_output=num_output, reuse=True)

        # Call tunable parameters
        theta = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="RSTNN")

    # Exponentially decayed learning rate
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(learning_rate=learning_rate, global_step=global_step,
                                               decay_steps=10000, decay_rate=0.96, staircase=True)

    # Adam optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss=training_loss, var_list=theta)

    # Load dataset
    train_eeg, train_label, valid_eeg, valid_label, test_eeg, test_label = utils.load_DATA(dataset_name=dataset_name, subject=subject,
                                                                                                 session_fold=session_fold)

    # Start training
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(keep_checkpoint_every_n_hours=24, max_to_keep=1000)

    print("Start Training, Network: " +network_name + ", Dataset: " + dataset_name + ", Subject ID: %02d, Session/Fold: %02d" %(subject, session_fold))

    total_batch = int(train_eeg.shape[0] / batch_size)
    performance = np.zeros(shape=(3, num_epochs))
    for epoch in range(num_epochs):
        # print("Operating Epoch %02d..." %epoch)
        # Randomize the training dataset for each epoch
        rand_idx = np.random.permutation(train_eeg.shape[0])

        train_eeg = train_eeg[rand_idx, :, :, :]
        train_label = train_label[rand_idx, :]

        train_loss = 0
        # Feed dictionaries
        for batch in range(total_batch):
            batch_X = train_eeg[batch * batch_size:(batch + 1) * batch_size, :, :, :]
            batch_Y = train_label[batch * batch_size:(batch + 1) * batch_size, :]

            tr_loss, tr_pred, _ = sess.run(fetches=[training_loss, train_prediction, optimizer],
                                           feed_dict={X_train: batch_X, Y_train: batch_Y})

        batch_X = train_eeg
        batch_Y = train_label
        val_pred = sess.run(fetches=[valid_prediction], feed_dict={X_valid: batch_X, Y_valid: batch_Y})
        val_pred = np.asarray(val_pred).astype(int)
        performance[0, epoch] = accuracy_score(y_true=np.squeeze(batch_Y), y_pred=np.squeeze(val_pred))


        batch_X = valid_eeg
        batch_Y = valid_label
        val_pred = sess.run(fetches=[valid_prediction], feed_dict={X_valid: batch_X, Y_valid: batch_Y})
        val_pred = np.asarray(val_pred).astype(int)
        performance[1, epoch] = accuracy_score(y_true=np.squeeze(batch_Y), y_pred=np.squeeze(val_pred))


        batch_X = test_eeg
        batch_Y = test_label
        tst_pred = sess.run(fetches=[test_prediction], feed_dict={X_test: batch_X, Y_test: batch_Y})
        tst_pred = np.asarray(tst_pred).astype(int)
        performance[2, epoch] = accuracy_score(y_true=np.squeeze(batch_Y), y_pred=np.squeeze(tst_pred))

    if print_detail == True:
        for i in range(performance.shape[-1]):
            print("%02dth epoch, Train: %.3f, Val: %.3f, Test: %.3f" %(i, performance[0, i], performance[1, i], performance[2, i]))

    tf.reset_default_graph()
    return

