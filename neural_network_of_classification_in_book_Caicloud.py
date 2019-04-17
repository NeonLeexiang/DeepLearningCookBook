# import list
import tensorflow as tf
from numpy.random import RandomState


"""
    :neonleexiang
    :2019-04-16
"""


BATCH_SIZE = 8
STEPS = 5000

# define the network layer, 2 layer w1[2, 3], w2[3, 1] output dimension=1, input dimension=2
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

# then we set the input output
x = tf.placeholder(tf.float32, shape=[None, 2], name='x')
y = tf.placeholder(tf.float32, shape=[None, 1], name='y')

# define the training process
a = tf.matmul(x, w1)
y_output = tf.matmul(a, w2)

# define the loss function
cross_entropy = -tf.reduce_mean(y * tf.log(tf.clip_by_value(y_output, 1e-10, 1.0)))

# define the training step
train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cross_entropy)

# now we begin the training
# first we create a random sample
rdm = RandomState(1)
DATASET_SIZE = 123
x_dataset = rdm.rand(DATASET_SIZE, 2)
y_dataset = [[int(x1 + x2 < 1)] for (x1, x2) in x_dataset]

# then we begin our tensorFlow training
# -------------------------------------------------------------------
sess = tf.Session()

init_op = tf.global_variables_initializer()

sess.run(init_op)
print(sess.run(w1))     # print to see the init of w1 and w2
print(sess.run(w2))

for epoch in range(STEPS):  # STEPS = 5000
    # each epoch we choose a batch to train
    start = (epoch * BATCH_SIZE) % DATASET_SIZE
    end = min(start + BATCH_SIZE, DATASET_SIZE)

    sess.run(train_step, feed_dict={x: x_dataset[start:end], y: y_dataset[start:end]})

    if epoch % 1000 == 0:
        # counting the cross entropy after a actual time
        total_cross_entropy = sess.run(cross_entropy, feed_dict={x: x_dataset, y: y_dataset})

        print("After {0} training steps, cross entropy on all data is: {1}".format(epoch, total_cross_entropy))

print(sess.run(w1))     # to check the w1, w2 update
print(sess.run(w2))

# end of the test

