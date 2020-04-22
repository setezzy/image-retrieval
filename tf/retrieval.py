from __future__ import division
import tensorflow as tf
import numpy as np
import pandas as pd
import argparse
import cv2
from img_trans import *
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# read the parameter
# argument parsing
parser = argparse.ArgumentParser(description='Main function for follow-up input generation and nc calculation')
parser.add_argument('query_batch_size', help="batch size of query images", type=int, default=1)
parser.add_argument('trans', help="tranformation type",
                    choices=['light', 'shift', 'rotate', 'zoom',
                             's_p_noise', 'dilation', 'erosion', 'draw_line',
                             ])

args = parser.parse_args()

def retrieval():
    """
    restore lenet model to conduct retrieval task
    """
    sess = tf.InteractiveSession()


    all_related = 0        # number of all related images
    retrieved_related = 0  # number of related images in retrieved ones
    retrieved = 0          # number of retrieved image


    # load meta graph
    saver = tf.train.import_meta_graph('./lenet5.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./'))
    print("Model Restored")


    # get tensors from graph
    graph = tf.get_default_graph()
    h4 = graph.get_tensor_by_name('Relu_3:0')
    x = graph.get_tensor_by_name('Placeholder:0')
    label = graph.get_tensor_by_name('Placeholder_1:0')
    keep_prob = graph.get_tensor_by_name('Placeholder_2:0')

    query_batch_size = args.query_batch_size
    batch = mnist.test.next_batch(query_batch_size)
    b = h4.eval(feed_dict={x: batch[0], label: batch[1], keep_prob: 1.0}) # feature extracted from fc layer
    test_digit = tf.argmax(batch[1], 1).eval() 

    precision = []
    recall = []

    for i in range(query_batch_size):
        # restore the test image to 28*28
        b1 = tf.reshape(batch[0], [query_batch_size, 28, 28]).eval()
        b2 = 255 * b1
        b3 = b2.astype(np.uint64)
        cv2.imwrite('./pics/source'+str(i)+'.jpg', b3[i])

        # =====================generate new image=============================
        #img = b3[i]
        #category = test_digit[i]
        #gen_image(i, img, args.trans, category)
        #print '%s finished' % str(trans)


        # ===========================retrieval task=================================

        # here I put in all test images for one batch
        batch_ = mnist.test.next_batch(10000)

        # extract the feature of fully-connected layer
        feat = h4.eval(feed_dict={x: batch_[0], label: batch_[1], keep_prob: 1.0})
        pred_digit = tf.argmax(batch_[1], 1).eval()

        # restore the image to 28*28
        feat1 = tf.reshape(batch_[0], [10000, 28, 28]).eval()
        feat2 = 255 * feat1
        feat3 = feat2.astype(np.uint64)

        index = []      # index of retrieved images
        distance = []   # distance between the feature vectors of test image and retrieved image

        for j in range(10000):
            if pred_digit[j] == test_digit[i]:
                all_related += 1

        for k in range(10000):
            # use cosine distance as similarity metric
            cos = np.dot(feat[k], b[i])/(np.linalg.norm(feat[k])*np.linalg.norm(b[i]))
            # regularize
            sim = 0.5+0.5*cos
            if sim >= 0.85:  # set a threshold
                index.append(k)
                similarity.append(sim)
                # write the retrieved images
                # cv2.imwrite('./pics/retrieved/'+str(k)+'.jpg', feat3[k])
                retrieved += 1
                if pred_digit[k] == test_digit[i]:
                    retrieved_related += 1

'''
def gen_image(i, img, trans, category):
    """
    # generate follow-up image
    :param i: image id
    :param img: source query image
    :param mr: image transformation type
    :param category: category of the image
    """
    img_source = img.copy()
    img_new = eval(trans)(img_source)
    path_follow = './pics/' + str(mr) + '/' + str(category)
    if not (os.path.exists(path_follow)):
        os.makedirs(path_follow)
    cv2.imwrite(path_follow + '/' + str(i) + '.jpg', img_new)
'''

if __name__ == '__main__':
    retrieval()
