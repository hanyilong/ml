import tensorflow as tf
import numpy as np

# Create 100 phony x, y data points in NumPy, y = x * 0.1 + 0.3
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3
logs_path = 'c:/tmp/tensorflow_logs/line/'
# Try to find values for W and b that compute y_data = W * x_data + b
# (We know that W should be 0.1 and b 0.3, but TensorFlow will
# figure that out for us.)
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

# Minimize the mean squared errors.
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# Before starting, initialize the variables.  We will 'run' this first.
init = tf.global_variables_initializer()
tf.summary.scalar("loss", loss)
merged_summary_op = tf.summary.merge_all()
# Launch the graph.
sess = tf.Session()
sess.run(init)
summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
# Fit the line.
for step in range(201):
    _, summary = sess.run([train,merged_summary_op])
    if step % 20 == 0:
        print(step, sess.run(W), sess.run(b))
        summary_writer.add_summary(summary, step)

# Learns best fit is W: [0.1], b: [0.3]