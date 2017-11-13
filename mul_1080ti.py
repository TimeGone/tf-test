import tensorflow as tf
import numpy as np

def swish(x):
    return  x * tf.nn.sigmoid(x)

def forfan(x):
    return x * tf.nn.tanh(x)

summaries_dir = '/tmp/mul_ubuntu'

activator = tf.nn.tanh
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)

# x_data = np.random.rand(64, 2) * 0.9 + 0.1
# y_data = x_data[:,:1] * x_data[:, 1:2]

x_test = np.random.rand(8192, 2) * 0.9 + 0.1
y_test = x_test[:,:1] * x_test[:,1:2]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
training = tf.placeholder(tf.bool, name='training')

W1 = tf.Variable(tf.random_normal([2, 10]), name='weight1')
b1 = tf.Variable(tf.random_normal([10]), name='bias1')
h1 = tf.matmul(X, W1) + b1
layer1 = activator(h1)
tf.summary.histogram('layer1', layer1)
# layer1 = activator(tf.contrib.layers.batch_norm(h1, is_training=training, fused=True))
# tf.summary.histogram('W1', W1)
# tf.summary.histogram('b1', b1)

W2 = tf.Variable(tf.random_normal([10, 5]), name='weight2')
b2 = tf.Variable(tf.random_normal([5]), name='bias2')
h2 = tf.matmul(layer1, W2) + b2
layer2 = activator(h2)
tf.summary.histogram('layer2', layer2)
# layer2 = activator(tf.contrib.layers.batch_norm(h2, is_training=training, fused=True))
# hypothesis = tf.matmul(layer1, W2) + b2
# tf.summary.histogram('W2', W2)
# tf.summary.histogram('b2', b2)

W3 = tf.Variable(tf.random_normal([5, 2]), name='weight3')
b3 = tf.Variable(tf.random_normal([2]), name='bias3')
h3 = tf.matmul(layer2, W3) + b3
layer3 = activator(h3)
tf.summary.histogram('layer3', layer3)
# tf.summary.histogram('W3', W3)

W4 = tf.Variable(tf.random_normal([2, 1]), name='weight4')
b4 = tf.Variable(tf.random_normal([1]), name='bias4')
hypothesis = tf.matmul(layer3, W4) + b4
tf.summary.histogram('hypothesis', hypothesis)

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
"""
for grad_n_var in grads_and_vars:
    tf.summary.histogram("{}-grad".format(grad_n_var[1].name), grad_n_var[0])
    tf.summary.histogram('{}-var'.format(grad_n_var[1].name), grad_n_var[1])
"""            
merged = tf.summary.merge_all()
saver = tf.train.Saver()

# config = tf.ConfigProto()
# config = tf.ConfigProto(device_count={"CPU": 12})
# config.intra_op_parallelism_threads = 16000
# config.inter_op_parallelism_threads = 16000
with tf.Session() as sess:
    train_writer = tf.summary.FileWriter(summaries_dir + '/1080ti', sess.graph)
    sess.run(tf.global_variables_initializer())
    for step in range(300000000):
#        if x_data.shape[0] > 64:
#            if step % 20 == 0:
#                np.random.shuffle(x_data)
#                x_data = np.delete(x_data, 1, axis=0)

        x_data = np.random.rand(256, 2) * 0.9 + 0.1
        y_data = x_data[:,:1] * x_data[:, 1:2]
        
        _, summary = sess.run([train, merged], feed_dict={X: x_data, Y: y_data, training:True})
        if step % 100 == 0:
            print(step, sess.run([loss, accuracy, relative_bias_mean], feed_dict={X:x_data, Y:y_data, training:False}), sess.run(relative_bias_mean, feed_dict={X:x_test, Y:y_test, training:False}), x_data.shape[0])
            if step >= 5000:
                train_writer.add_summary(summary, step)

        if step % 5000 == 0:
            saver.save(sess, 'ckpt/mul_1080ti.ckpt', global_step = step)
            
    x_test = np.random.rand(8192, 2) * 0.9 + 0.1
    y_test = x_test[:,:1] * x_test[:,1:2] 
    
    h, a = sess.run([hypothesis, accuracy], feed_dict={X:x_test, Y:y_test, training:False})
    print("\nHypothesis: ", h , "\ny_value", y_test, "\nAccuracy: ", a)
