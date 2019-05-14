from __future__ import division
import cv2
import os
import argparse
import tensorflow as tf
import pandas as pd
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# argument parsing
parser = argparse.ArgumentParser(description='Main function for follow-up input generation and nc calculation')
parser.add_argument('mr', choices=['rotate', 'erosion', 'dilation'])
args = parser.parse_args()

# restore lenet model to conduct retrieval task
sess = tf.InteractiveSession()
# load meta graph
saver = tf.train.import_meta_graph('./lenet5.ckpt.meta')
saver.restore(sess, tf.train.latest_checkpoint('./'))
print "Model Restored"

# get tensors from graph
graph = tf.get_default_graph()
# fclayer 1
f6 = graph.get_tensor_by_name('Relu_3:0')

x = graph.get_tensor_by_name('Placeholder:0')
label = graph.get_tensor_by_name('Placeholder_1:0')
keep_prob = graph.get_tensor_by_name('Placeholder_2:0')


def read_from_disk():
    """
    read file names and labels
    :return:
        file_list: file name list
        label_list: label list
    """

    file_list = []
    label_list = []
    name_list = []
    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    for class_item in classes:
        dir_name = './pics/'+args.mr+'/'+class_item
        for files in os.listdir(dir_name):
            file_list.append(dir_name+'/'+files)
            label_list.append(classes.index(class_item))
            name_list.append(files)
    return file_list, label_list, name_list


def image_init(img_path):
    """
    image preprocessing
    :param img_path: the path of query image
    :return: image tensor
    """
    im = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    x_img = im/float(255)
    x_img = np.reshape(x_img, [-1, 784])
    return x_img


def label_init(y):
    """
    transfer label to one-hot representation
    :param y: label list
    :return: one-hot tensor
    """
    batch_size = tf.size(y)
    label_list = tf.expand_dims(y, 1)
    indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
    concated = tf.concat([indices, label_list], 1)
    label_list = tf.sparse_to_dense(concated, tf.stack([batch_size, 10]), 1.0, 0.0)
    return label_list


def retrieval():

    """
    retrieval task on MNIST
    """

    all_related = 0
    retrieved_related = 0
    retrieved = 0
    precision = []
    recall = []
    neuron_coverage = []
    indexes = []

    file_list, label_list, name_list = read_from_disk()
    label_list = label_init(label_list)
    
    # here I put in all test images for one batch
    batch_ = mnist.test.next_batch(10000, shuffle=False)
    # extract the feature of fully-connected layer
    feat = f6.eval(feed_dict={x: batch_[0], label: batch_[1], keep_prob: 1.0})
    # label of retrieved image
    pred_digit = tf.argmax(batch_[1], 1).eval()

    for i in range(0, 1000):

        # read query image
        img_path = file_list[i]
        x_img = image_init(img_path)
        img_name = name_list[i]
        img_index = int(img_name.split('.')[0])
        indexes.append(img_index)

        # read label
        labels = sess.run(label_list)
        y_label = labels[i]
        y_label = np.reshape(y_label, [-1, 10])

        query_feat = f6.eval(feed_dict={x: x_img, label: y_label, keep_prob: 1.0})
        query_digit = tf.argmax(y_label, 1).eval()


        # ===========================retrieval task=================================

        index = []  
        similarity = []  
        pred_label = []

        for j in xrange(10000):
            if pred_digit[j] == query_digit:
                all_related += 1

        for m in xrange(10000):
            cos = np.dot(feat[m], query_feat[0])/(np.linalg.norm(feat[m])*np.linalg.norm(query_feat[0]))
            sim = 0.5+0.5*cos
            if sim >= 0.85:  # set a thresholds
                index.append(m)
                similarity.append(sim)
                pred_label.append(pred_digit[m])
                retrieved += 1
                if pred_digit[m] == query_digit:
                    retrieved_related += 1

        df = pd.DataFrame({'retrieved_image_index': index, 'similarity': similarity, 'label': pred_label})

        # define the evaluation metric
        r = retrieved_related / all_related
        p = retrieved_related / retrieved
        f = r * p * 2 / (r + p)
        precision.append(p)
        recall.append(r)

    # write precision and recall
    df_measure = pd.DataFrame({'index': indexes,  'precision': precision, 'recall': recall})
    df_measure = df_measure.sort_values('index', ascending=True)
    df_measure.to_csv('./' + args.mr + '/' + str(i) + '.csv', index=True)


if __name__ == '__main__':
    retrieval()
