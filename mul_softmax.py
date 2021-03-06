import tensorflow as tf
import numpy as np

def swish(x):
    return  x * tf.nn.sigmoid(x)

def forfan(x):
    return x * tf.nn.tanh(x)

def get_label_index(value, num_classes):
    indexs = (np.log(value/lower_bound) / np.log(1+threshold)).astype(int)
    indexs[indexs < 0] = 0
    return indexs[:,0]

def get_label(value, num_classes):
    labels = np.zeros((value.shape[0], num_classes))
    indexs = get_label_index(value, num_classes)
    labels[range(value.shape[0]), indexs] = 1
    return labels
#   another way: return np.eye(num_classes)[indexs]

summaries_dir = '/tmp/mul_ubuntu'

activator = tf.nn.tanh
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)

lower_bound = 0.01 # for x1 and x2 > 0.1
threshold = 0.01 # relative bia shoud less than it
num_clips = int(np.log(1/lower_bound) / np.log(1+threshold)) + 1

X = tf.placeholder(tf.float32)
Y_label = tf.placeholder(tf.float32, [None, num_clips])
training = tf.placeholder(tf.bool, name='training')

W1 = tf.Variable(tf.random_normal([2, 1000]), name='weight1')
b1 = tf.Variable(tf.random_normal([1000]), name='bias1')
h1 = tf.matmul(X, W1) + b1
layer1 = activator(h1)
# layer1 = activator(tf.contrib.layers.batch_norm(h1, is_training=training, fused=True))
# tf.summary.histogram('layer1', layer1)
W2 = tf.Variable(tf.random_normal([1000, 5000]), name='weight2')
b2 = tf.Variable(tf.random_normal([5000]), name='bias2')
h2 = tf.matmul(layer1, W2) + b2
layer2 = activator(h2)

W3 = tf.Variable(tf.random_normal([5000, num_clips]), name='weight3')
b3 = tf.Variable(tf.random_normal([num_clips]), name='bias3')
h3 = tf.matmul(layer2, W3) + b3
predicts = tf.nn.softmax(h3)
# layer2 = activator(tf.contrib.layers.batch_norm(h2, is_training=training, fused=True))
# tf.summary.histogram('predicts', predicts)

correct = tf.equal(tf.argmax(predicts, 1), tf.argmax(Y_label, 1))
accuracy = tf.reduce_mean(tf.cast(correct, dtype=tf.float32))
tf.summary.scalar('accuracy', accuracy)
loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=h3, labels=Y_label))
tf.summary.scalar('loss', loss)

grads_and_vars = optimizer.compute_gradients(loss)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train = optimizer.apply_gradients(grads_and_vars)

# for grad_n_var in grads_and_vars:
#     tf.summary.histogram("{}-grad".format(grad_n_var[1].name), grad_n_var[0])
#     tf.summary.histogram('{}-var'.format(grad_n_var[1].name), grad_n_var[1])
            
merged = tf.summary.merge_all()
saver = tf.train.Saver()
# config = tf.ConfigProto()
# config = tf.ConfigProto(device_count={"CPU": 12})
# config.intra_op_parallelism_threads = 160
# config.inter_op_parallelism_threads = 160

x_test = np.random.rand(8192, 2)
y_test = x_test[:,:1] * x_test[:,1:2]
y_label_test = get_label(y_test, num_clips)

with tf.Session() as sess:
    train_writer = tf.summary.FileWriter(summaries_dir + '/softmax', sess.graph)
    sess.run(tf.global_variables_initializer())
    for step in range(30000000):
        x_data = np.random.rand(1024, 2)
        y_data = x_data[:,:1] * x_data[:, 1:2]
        y_label_data = get_label(y_data, num_clips)
        _, summary = sess.run([train, merged], feed_dict={X: x_data, Y_label: y_label_data, training:True})
        if step % 250 == 0:
            print(step, sess.run([loss, accuracy], feed_dict={X:x_data, Y_label:y_label_data, training:False}), sess.run(accuracy, feed_dict={X:x_test, Y_label:y_label_test, training:False}))
            if step >= 5000:
                train_writer.add_summary(summary, step)
        if step % 5000 == 0:
            saver.save(sess, 'ckpt/mul_softmax.ckpt', global_step = step)
            
    x_test = np.random.rand(8192, 2)
    y_test = x_test[:,:1] * x_test[:,1:2]
    y_label_test = get_label(y_test, num_clips) 
    p, a = sess.run([predicts, accuracy], feed_dict={X:x_test, Y_label:y_label_test, training:False})
    print("\nPredict: ", p , "\ny_value", y_test, "\ny_lable", y_label_test, "\nAccuracy: ", a)
