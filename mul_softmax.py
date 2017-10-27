import tensorflow as tf
import numpy as np

def swish(x):
    return  x * tf.nn.sigmoid(x)

def forfan(x):
    return x * tf.nn.tanh(x)

def get_label(value, class_num):
    labels = np.zeros((value.shape[0], class_num))
    indexs = (value * class_num).astype(int)
    indexs = indexs[:,0]
    labels[range(value.shape[0]), indexs] = 1
    return labels

summaries_dir = '/tmp/mul_ubuntu'

activator = swish
optimizer = tf.train.AdamOptimizer(learning_rate=0.00001)

x_data = np.random.rand(5000, 2)
y_data = x_data[:,:1] * x_data[:, 1:2]
y_label_data = get_label(y_data, 100)
# print(y_data)
# print(y_label_data)

x_test = np.random.rand(8192, 2)
y_test = x_test[:,:1] * x_test[:,1:2]
y_label_test = get_label(y_test, 100)

X = tf.placeholder(tf.float32)
Y_label = tf.placeholder(tf.float32)
training = tf.placeholder(tf.bool, name='training')

W1 = tf.Variable(tf.random_normal([2, 200]), name='weight1')
b1 = tf.Variable(tf.random_normal([200]), name='bias1')
h1 = tf.matmul(X, W1) + b1
layer1 = activator(h1)
# layer1 = activator(tf.contrib.layers.batch_norm(h1, is_training=training, fused=True))
# tf.summary.histogram('layer1', layer1)

W2 = tf.Variable(tf.random_normal([200, 100]), name='weight2')
b2 = tf.Variable(tf.random_normal([100]), name='bias2')
h2 = tf.matmul(layer1, W2) + b2
predict = tf.nn.softmax(h2)
# layer2 = activator(tf.contrib.layers.batch_norm(h2, is_training=training, fused=True))
# tf.summary.histogram('predict', predict)

correct = tf.equal(tf.argmax(predict), tf.argmax(Y_label))
accuracy = tf.reduce_mean(tf.cast(correct, dtype=tf.float32))
tf.summary.scalar('accuracy', accuracy)
# loss = tf.reduce_sum(tf.square(tf.subtract(hypothesis, Y)))
loss = -tf.reduce_sum(Y_label * tf.log(predict))
tf.summary.scalar('loss', loss)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

grads_and_vars = optimizer.compute_gradients(loss)
with tf.control_dependencies(update_ops):
    train = optimizer.apply_gradients(grads_and_vars)

# for grad_n_var in grads_and_vars:
#     tf.summary.histogram("{}-grad".format(grad_n_var[1].name), grad_n_var[0])
#     tf.summary.histogram('{}-var'.format(grad_n_var[1].name), grad_n_var[1])
            
merged = tf.summary.merge_all()

config = tf.ConfigProto()
# config = tf.ConfigProto(device_count={"CPU": 12})
config.intra_op_parallelism_threads = 160
config.inter_op_parallelism_threads = 160
with tf.Session(config=config) as sess:
    train_writer = tf.summary.FileWriter(summaries_dir + '/softmax', sess.graph)
    sess.run(tf.global_variables_initializer())
    for step in range(80000):
        # x_data = np.random.rand(256, 2)
        # y_data = x_data[:,:1] * x_data[:, 1:2]
        # y_label_data = get_label(y_data, 100)
        _, summary = sess.run([train, merged], feed_dict={X: x_data, Y_label: y_label_data, training:True})
        if step % 100 == 0:
            print(step, sess.run([loss, accuracy], feed_dict={X:x_data, Y_label:y_label_data, training:False}), sess.run(accuracy, feed_dict={X:x_test, Y_label:y_label_test, training:False}), x_data.shape[0])
            if step >= 5000:
                train_writer.add_summary(summary, step)

    x_test = np.random.rand(8192, 2)
    y_test = x_test[:,:1] * x_test[:,1:2]
    y_label_test = get_label(y_test, 100) 
    
    p, a = sess.run([predict, accuracy], feed_dict={X:x_test, Y_label:y_label_test, training:False})
    print("\nPredict: ", p , "\ny_value", y_test, "\ny_lable", y_label_test, "\nAccuracy: ", a)