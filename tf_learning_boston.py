import tensorflow as tf

# Global parameters
DATA_FILE = "boston_housing.csv"
BATCH_SIZE = 10     # batch means 一批；一次所制之量，分批处理； 批量，批处理
NUM_FEATURES = 14


# ----------------------------- data preprocess ---------------------------------
def data_generator(filename):
    """

    generates tensor in batches of size BATCH_SIZE.
    args: string tensor

    :param filename: from which data is to be read
    :return: tensors feature_batch and label_batch
    """
    f_queue = tf.train.string_input_producer(filename)
    reader = tf.TextLineReader(skip_header_lines=1)     # skip the first
    _, value = reader.read(f_queue)

    record_defaults = [[0.0] for _ in range(NUM_FEATURES)]

    data = tf.decode_csv(value, record_defaults=record_defaults)
    features = tf.stack(tf.gather_nd(data, [[5], [10], [12]]))  # choose feature RM, PTRATIO, LSTAT
    label = data[-1]

    # minimum number elements in the queue after a
    min_after_dequeue = 10 * BATCH_SIZE

    # the maximum number of elements in the queue
    capacity = 20 * BATCH_SIZE

    # shuffle the data to generate BATCH_SIZE sample pairs
    feature_batch, label_batch = tf.train.shuffle_batch([features, label], batch_size=BATCH_SIZE, capacity=capacity,
                                                        min_after_dequeue=min_after_dequeue)

    return feature_batch, label_batch


def generate_data(feature_batch, label_batch):
    """

    :param feature_batch:
    :param label_batch:
    :return:
    """
    with tf.Session() as sess:
        # initialize the queue threads      # Coordinate：坐标；同等的人或物；并列的，同等的
        coord = tf.train.Coordinator()      # Coordinator means 协调器，协调员
        threads = tf.train.start_queue_runners(coord=coord)
        for _ in range(5):  # generate 5 batches
            features, labels = sess.run([feature_batch, label_batch])
            print(features, 'HI')

        coord.request_stop()
        coord.join(threads)

# -------------------------------- end of data preprocess ------------------------------------


if __name__ == '__main__':
    _feature_batch, _label_batch = data_generator([DATA_FILE])
    generate_data(_feature_batch, _label_batch)

    """
    there are 16 lines of data's MEDV equal to 50.0, and they are useless, 
    so we can use some option to delete them:
    
    condition = tf.equal(data[13], tf.constant(50.0))
    data = tf.where(condition, tf.zeros(NUM_FEATURES), data[:])  # to change the data into zeros
    """
