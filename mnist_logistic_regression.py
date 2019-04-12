# import list
import tensorflow as tf
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
from tensorflow.examples.tutorials.mnist import input_data


"""
    :neonleexiang
    :2019-04-12
    to try some complex logistic regression with dataset: Minst
"""

MAX_EPOCHS = 30
BATCH_SIZE = 10


def logistic_regression_with_mnist():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  # use one_hot encoding

    # [55000, 784]
    x_placeholder = tf.placeholder(tf.float32, [None, 784], name="X")
    y_placeholder = tf.placeholder(tf.float32, [None, 10], name="Y")

    w = tf.Variable(tf.zeros([784, 10]), name='W')
    b = tf.Variable(tf.zeros([10]), name='b')

    # create logistic regression model
    with tf.name_scope("wx_b") as scope:
        y_hat = tf.nn.softmax(tf.matmul(x_placeholder, w) + b)

        # create summary histogram information of w and b
        w_h = tf.summary.histogram("Weight", w)
        b_h = tf.summary.histogram("Biases", b)

    # define cross-entropy and loss function
    # also we add the name scope and summary to make model more visible
    # use scalar summary to attain the loss function which changing at time
    with tf.name_scope('cross-entropy') as scope:
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_placeholder, logits=y_hat))
        tf.summary.scalar('cross-entropy', loss)

    with tf.name_scope('Train') as scope:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

    # initialize
    init_option = tf.global_variables_initializer()

    # merge all the summary
    merged_summary_option = tf.summary.merge_all()

    # then we run
    with tf.Session() as sess:
        sess.run(init_option)   # initialize

        summary_writer = tf.summary.FileWriter('graphs_of_mnist_logistic_regression', sess.graph)

        # training
        for epoch in range(MAX_EPOCHS):
            loss_avg = 0

            num_of_batch = int(mnist.train.num_examples / BATCH_SIZE)

            for i in range(num_of_batch):
                batch_xs, batch_ys = mnist.train.next_batch(100)    # get the nex batch of data

                # run the optimizer
                _, loss_record, summary_str = sess.run([optimizer, loss, merged_summary_option],
                                                       feed_dict={x_placeholder: batch_xs, y_placeholder: batch_ys})
                loss_avg += 1

                # add all summaries per batch
                summary_writer.add_summary(summary_str, epoch*num_of_batch+i)

            loss_avg = loss_avg / num_of_batch
            print('Epoch {0}: Loss- {1}'.format(epoch, loss_avg))

        print("Done")
        # print(sess.run(accuracy, feed_dict={x_placeholder: mnist.test.images, y_placeholder: mnist.test.labels}))

                   
if __name__ == '__main__':
    logistic_regression_with_mnist()
