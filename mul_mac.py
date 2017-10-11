import tensorflow as tf
import numpy as np

#x_data = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 1.0]], dtype=np.float32)
#y_data = np.array([[0.02], [0.12], [0.3], [0.56], [0.9]], dtype=np.float32)
x_data = np.random.rand(50000, 2)
y_data = x_data[:,:1] * x_data[:, 1:2]

#x_test_data = np.array([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4], [0.5, 0.5]], dtype=np.float32)
#y_test_data = np.array([[0.01], [0.04], [0.09], [0.16], [0.25]], dtype=np.float32)
x_test_data = np.random.rand(5000, 2)
y_test_data = x_test_data[:,:1] * x_test_data[:,1:2]
print(y_test_data)

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_normal([2, 8]), name='weight1')
b1 = tf.Variable(tf.random_normal([8]), name='bias1')
#layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)
layer1 = tf.nn.relu(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_normal([8, 4]), name='weight2')
b2 = tf.Variable(tf.random_normal([4]), name='bias2')
#layer2 = tf.sigmoid(tf.matmul(layer1, W2) + b2)
layer2 = tf.nn.relu(tf.matmul(layer1, W2) + b2)

W3 = tf.Variable(tf.random_normal([4, 1]), name='weight3')
b3 = tf.Variable(tf.random_normal([1]), name='bias3')
#hypothesis = tf.sigmoid(tf.matmul(layer2, W3) + b3)
hypothesis = tf.matmul(layer2, W3) + b3

# cost/loss function
# cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
cost = tf.reduce_sum(tf.square(tf.subtract(hypothesis, Y)))
train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(cost)
# Accuracy computation
# True if hypothesis>0.5 else False
accuracy = tf.reduce_mean(tf.cast(tf.div(tf.abs(tf.subtract(hypothesis, Y)), Y) < 0.01, dtype=tf.float32))
# Launch graph
with tf.Session() as sess:
   # Initialize TensorFlow variables
   sess.run(tf.global_variables_initializer())
   for step in range(200001):
       #x_data = np.random.rand(256, 2)
       #y_data = x_data[:,:1] * x_data[:, 1:2]
       sess.run(train, feed_dict={X: x_data, Y: y_data})
       if step % 500 == 0:
           print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}))

   # Accuracy report
   h, a = sess.run([hypothesis, accuracy],
                      feed_dict={X: x_test_data, Y: y_test_data})
   #print("\nY_test: ", y_test_data, "\nHypothesis: ", h, "\nAccuracy: ", a)
   #print(sess.run([W1, W2]))
   print("\nAccuracy: ", a)

