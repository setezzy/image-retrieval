import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import lenet5_model

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


def train():
    """
    train the lenet model on MNIST
    """

    sess = tf.InteractiveSession()

    x = tf.placeholder(tf.float32, shape=[None, 784])
    label = tf.placeholder(tf.float32, shape=[None, 10])
    keep_prob = tf.placeholder(tf.float32)

    l_ = lenet5_model.model(x, keep_prob)
    cross_entropy = -tf.reduce_sum(label * tf.log(l_))
    # minimize the cross entropy with lr=0.0001
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_predict = tf.equal(tf.argmax(l_, 1), tf.argmax(label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))

    # create summary operations in tensorboard
    tf.summary.scalar('loss', cross_entropy)
    tf.summary.scalar('accuracy', accuracy)

    sess.run(tf.global_variables_initializer())

    # merge all the summary nodes
    merged = tf.summary.merge_all()
    # write data to local file
    summary_writer = tf.summary.FileWriter('./mnistEnv/', graph=sess.graph)

    for i in range(20000):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            # print the log every 100 steps
            train_accuracy = accuracy.eval(
                feed_dict={
                    x: batch[0],
                    label: batch[1],
                    keep_prob: 1.0
                })
            print("step %d, training accuracy %.4f" % (i, train_accuracy))
        train_step.run(feed_dict={
            x: batch[0],
            label: batch[1],
            keep_prob: 0.5
        })

        # run the merged node
        # my tensorboard at http:// zhang:6006
        summary = sess.run(merged, feed_dict={x: batch[0], label: batch[1], keep_prob: 1.0})
        summary_writer.add_summary(summary, i)

    # save the model
    saver = tf.train.Saver()
    save_path = saver.save(sess, "/home/zzy/py/lenet5/lenet5.ckpt")
    print("Model Saved in File: ", save_path)

    sess.close()


if __name__ == '__main__':
    train()
