import tensorflow as tf
import numpy as np

summaries_dir = '/tmp/mul_ubuntu'

activator = tf.nn.relu
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

x_data = np.random.rand(50000, 2) * 0.1
y_data = x_data[:,:1] * x_data[:, 1:2]

x_test = np.random.rand(5000, 2) * 0.1
y_test = x_test[:,:1] * x_test[:,1:2]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_normal([2, 6]), name='weight1')
b1 = tf.Variable(tf.random_normal([6]), name='bias1')
layer1 = activator(tf.matmul(X, W1) + b1)
tf.summary.histogram('W1', W1)

W2 = tf.Variable(tf.random_normal([6, 4]), name='weight2')
b2 = tf.Variable(tf.random_normal([4]), name='bias2')
layer2 = activator(tf.matmul(layer1, W2) + b2)
tf.summary.histogram('W2', W2)

W3 = tf.Variable(tf.random_normal([4, 1]), name='weight3')
b3 = tf.Variable(tf.random_normal([1]), name='bias3')
hypothesis = tf.matmul(layer2, W3) + b3
tf.summary.histogram('W3', W3)

relative_bias = tf.div(tf.abs(tf.subtract(hypothesis, Y)), Y)
relative_bias_mean = tf.reduce_mean(relative_bias)
accuracy = tf.reduce_mean(tf.cast(relative_bias < 0.01, dtype=tf.float32))
loss = tf.reduce_sum(tf.div(tf.square(tf.subtract(hypothesis, Y)), Y))
tf.summary.scalar('loss', loss)

grads_and_vars = optimizer.compute_gradients(loss)
train = optimizer.apply_gradients(grads_and_vars)
for grad_n_var in grads_and_vars:
    tf.summary.histogram("{}-grad".format(grad_n_var[1].name), grad_n_var[0])
    tf.summary.histogram('{}-var'.format(grad_n_var[1].name), grad_n_var[1])
#for index, grad in enumerate(grads_and_vars):
#    tf.summary.histogram("{}-grad".format(grads_and_vars[index][1].name), grads_and_vars[index][0])
            
merged = tf.summary.merge_all()

with tf.Session() as sess:
    train_writer = tf.summary.FileWriter(summaries_dir + '/train', sess.graph)
    sess.run(tf.global_variables_initializer())
    for step in range(400001):
#        x_data = np.concatenate((x_data, np.random.rand(256, 2)), axis=0)
        x_data[0:256] = np.random.rand(256, 2) * 0.1
        y_data = x_data[:,:1] * x_data[:, 1:2]

        _, summary = sess.run([train, merged], feed_dict={X: x_data, Y: y_data})
        if step % 250 == 0:
            print(step, sess.run([loss, grads_and_vars[1]], feed_dict={X:x_data, Y:y_data}), sess.run([relative_bias, accuracy], feed_dict={X:x_test, Y:y_test}))
            if step >= 5000:
                train_writer.add_summary(summary, step)

    x_test = np.random.rand(5000, 2) * 0.1
    y_test = x_test[:,:1] * x_test[:,1:2] 

    h, a = sess.run([hypothesis, accuracy], feed_dict={X:x_test, Y:y_test})
    print("\nHypothesis: ", h , "\ny_value", y_test, "\nAccuracy: ", a)
