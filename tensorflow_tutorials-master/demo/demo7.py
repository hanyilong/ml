import tensorflow as tf 

def my_fun(arg):
    arg = tf.convert_to_tensor(arg, dtype=tf.float32)
    return tf.multiply(arg, arg) + arg

value_1 = my_fun([[1.0,2.0],[3.0,4.0]])
r = tf.random_normal([2,3], mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
r1 = tf.truncated_normal([2,3], mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
r2 = tf.random_uniform([2,3], minval=0, maxval=None, dtype=tf.float32, seed=None, name=None)
with tf.Session() as sess:
    print(value_1.eval())
    print(r.eval())
    print(r1.eval())
    print(r2.eval())


x = tf.constant([[[ 1,  2,  3, 4],
                  [ 5,  6,  7, 8],
                  [ 9,  10,  11, 12]],
                 [[ 13,  14,  15, 16],
                  [ 17,  18,  19, 20],
                  [ 21,  22,  23, 24]]])

with tf.Session() as sess:
    print(x.shape)
    print(x.eval())
    y = tf.transpose(x)
    print(y.shape)
    print(tf.transpose(x).eval())
    y = tf.transpose(x,perm = [0,2,1])
    print(y.shape)
    print(tf.transpose(x).eval())
print(x)