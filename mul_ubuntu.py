import tensorflow as tf
import numpy as np

def swish(x):
    return  x * tf.nn.sigmoid(x)

def forfan(x):
    return x * tf.nn.tanh(x)

summaries_dir = '/tmp/mul_ubuntu'

activator = tf.nn.tanh
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

x_data = np.random.rand(65536, 2)
y_data = x_data[:,:1] * x_data[:, 1:2]

x_test = np.random.rand(8192, 2)
y_test = x_test[:,:1] * x_test[:,1:2]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
training = tf.placeholder(tf.bool, name='training')

W1 = tf.Variable(tf.random_normal([2, 200]), name='weight1')
b1 = tf.Variable(tf.random_normal([200]), name='bias1')
h1 = tf.matmul(X, W1) + b1
layer1 = activator(h1)
# layer1 = activator(tf.contrib.layers.batch_norm(h1, is_training=training, fused=True))
# tf.summary.histogram('W1', W1)
# tf.summary.histogram('b1', b1)

W2 = tf.Variable(tf.random_normal([200, 100]), name='weight2')
b2 = tf.Variable(tf.random_normal([100]), name='bias2')
h2 = tf.matmul(layer1, W2) + b2
layer2 = activator(h2)
# layer2 = activator(tf.contrib.layers.batch_norm(h2, is_training=training, fused=True))
# hypothesis = tf.matmul(layer1, W2) + b2
# tf.summary.histogram('W2', W2)
# tf.summary.histogram('b2', b2)

W3 = tf.Variable(tf.random_normal([100, 1]), name='weight3')
b3 = tf.Variable(tf.random_normal([1]), name='bias3')
hypothesis = tf.matmul(layer2, W3) + b3
# tf.summary.histogram('W3', W3)

relative_bias = tf.div(tf.abs(tf.subtract(hypothesis, Y)), Y)
relative_bias_mean = tf.reduce_mean(relative_bias)
tf.summary.scalar('relative_bias_mean', relative_bias_mean)
accuracy = tf.reduce_mean(tf.cast(relative_bias < 0.01, dtype=tf.float32))
loss = tf.reduce_sum(tf.square(tf.subtract(hypothesis, Y)))
tf.summary.scalar('loss', loss)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

grads_and_vars = optimizer.compute_gradients(loss)
with tf.control_dependencies(update_ops):
    train = optimizer.apply_gradients(grads_and_vars)

for grad_n_var in grads_and_vars:
    tf.summary.histogram("{}-grad".format(grad_n_var[1].name), grad_n_var[0])
    tf.summary.histogram('{}-var'.format(grad_n_var[1].name), grad_n_var[1])
#for index, grad in enumerate(grads_and_vars):
#    tf.summary.histogram("{}-grad".format(grads_and_vars[index][1].name), grads_and_vars[index][0])
            
merged = tf.summary.merge_all()

config = tf.ConfigProto()
# config = tf.ConfigProto(device_count={"CPU": 12})
config.intra_op_parallelism_threads = 160
config.inter_op_parallelism_threads = 160
with tf.Session(config=config) as sess:
    train_writer = tf.summary.FileWriter(summaries_dir + '/train', sess.graph)
    sess.run(tf.global_variables_initializer())
    for step in range(800000):
#        x_data = np.concatenate((x_data, np.random.rand(256, 2)), axis=0)
        if x_data.shape[0] > 256:
            if step % 4 == 0:
                np.random.shuffle(x_data)
                x_data = np.delete(x_data, 1, axis=0)
        x_data[0:256] = np.random.rand(256, 2)
        y_data = x_data[:,:1] * x_data[:, 1:2]
        
        _, summary = sess.run([train, merged], feed_dict={X: x_data, Y: y_data, training:True})
        if step % 100 == 0:
            print(step, sess.run([loss, accuracy, relative_bias_mean], feed_dict={X:x_data, Y:y_data, training:False}), sess.run(relative_bias_mean, feed_dict={X:x_test, Y:y_test, training:False}), x_data.shape[0])
            if step >= 5000:
                train_writer.add_summary(summary, step)

    x_test = np.random.rand(8192, 2)
    y_test = x_test[:,:1] * x_test[:,1:2] 
    
    h, a = sess.run([hypothesis, accuracy], feed_dict={X:x_test, Y:y_test, training:False})
    print("\nHypothesis: ", h , "\ny_value", y_test, "\nAccuracy: ", a)
