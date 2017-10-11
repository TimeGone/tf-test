import tensorflow as tf
import numpy as np

x_data = np.random.rand(50000, 2) * 0.6 + 0.2
y_data = x_data[:,:1] * x_data[:, 1:2]

x_test = np.random.rand(5000, 2) * 0.6 + 0.2
y_test = x_test[:,:1] * x_test[:,1:2]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_normal([2, 20]), name='weight1')
b1 = tf.Variable(tf.random_normal([20]), name='bias1')
layer1 = tf.nn.tanh(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_normal([20, 1]), name='weight2')
b2 = tf.Variable(tf.random_normal([1]), name='bias2')
hypothesis = tf.nn.tanh(tf.matmul(layer1, W2) + b2)
'''
W3 = tf.Variable(tf.random_normal([4, 1]), name='weight3')
b3 = tf.Variable(tf.random_normal([1]), name='bias3')
hypothesis = tf.matmul(layer2, W3) + b3
'''
cost = tf.reduce_sum(tf.square(tf.subtract(hypothesis, Y)))
train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

accuracy = tf.reduce_mean(tf.cast(tf.div(tf.abs(tf.subtract(hypothesis, Y)), Y) < 0.01, dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(400001):
#        x_data = np.concatenate((x_data, np.random.rand(256, 2)), axis=0)
        x_data[0:256] = np.random.rand(256, 2) * 0.6 + 0.2
        y_data = x_data[:,:1] * x_data[:, 1:2]

        sess.run(train, feed_dict={X: x_data, Y: y_data})
        if step % 250 == 0:
            print(step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(accuracy, feed_dict={X:x_test, Y:y_test}))

    x_test = np.random.rand(5000, 2)
    y_test = x_test[:,:1] * x_test[:,1:2]
    h, a = sess.run([hypothesis, accuracy], feed_dict={X:x_test, Y:y_test})
    print("\nHypothesis: ", h , "\ny_value", y_test, "\nAccuracy: ", a)
