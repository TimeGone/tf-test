import tensorflow as tf
import numpy as np

x_data = np.random.rand(50000, 2)
y_data = x_data[:, 0:1] * x_data[:, 1:2]

x_test = np.random.rand(1000, 2)
y_test = x_test[:, 0:1] * x_test[:, 1:2]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_normal([2, 10]), name = 'weight1')
b1 = tf.Variable(tf.random_normal([10]), name = 'bias1')
layer1 = tf.nn.sigmoid(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_normal([10, 6]), name = 'weight2')
b2 = tf.Variable(tf.random_normal([6]), name = 'bias2')
layer2 = tf.nn.sigmoid(tf.matmul(layer1, W2) + b2)

W3 = tf.Variable(tf.random_normal([6, 4]), name = 'weight3')
b3 = tf.Variable(tf.random_normal([4]), name = 'bias3')
layer3 = tf.nn.sigmoid(tf.matmul(layer2, W3) + b3)

W4 = tf.Variable(tf.random_normal([4, 1]), name = 'weight4')
b4 = tf.Variable(tf.random_normal([1]), name = 'bias4')
hypothesis = tf.nn.sigmoid(tf.matmul(layer3, W4) + b4)

#cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
#cost = tf.reduce_sum(tf.square(tf.subtract(hypothesis, Y)))
cost = tf.reduce_sum(tf.div(tf.square(tf.subtract(hypothesis, Y)), Y))
train = tf.train.AdamOptimizer(learning_rate = 0.1).minimize(cost)

accuracy = tf.reduce_mean(tf.cast(tf.div(tf.abs(tf.subtract(hypothesis, Y)), Y) < 0.01, dtype = tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(2000001):
        x_data = np.random.rand(50000, 2)
        y_data = x_data[:, 0:1] * x_data[:, 1:2]

        sess.run(train, feed_dict={X: x_data, Y: y_data})
        if step % 250 == 0:
            print(step, sess.run([cost, accuracy], feed_dict={X:x_data, Y:y_data}))

    h, a = sess.run([hypothesis, accuracy], feed_dict={X:x_test, Y:y_test})
    print("\nHypothesis: ", h , "\ny_value", y_test, "\nAccuracy: ", a)
