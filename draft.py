import tensorflow as tf
import numpy as np

with tf.name_scope('placeholders'):
    x = tf.placeholder('float', [None, 1])
    y = tf.placeholder('float', [None, 1])

with tf.name_scope('neural_network'):
    x1 = tf.contrib.layers.fully_connected(x, 100)
    x2 = tf.contrib.layers.fully_connected(x1, 100)
    result = tf.contrib.layers.fully_connected(x2, 1,
                                               activation_fn=None)

    loss = tf.nn.l2_loss(result - y)

with tf.name_scope('optimizer'):
    train_op = tf.train.AdamOptimizer().minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Train the network
    for i in range(10000):
        xpts = np.random.rand(100) * 10
        ypts = np.sin(xpts)

        _, loss_result = sess.run([train_op, loss],
                                  feed_dict={x: xpts[:, None],
                                             y: ypts[:, None]})

        print('iteration {}, loss={}'.format(i, loss_result))