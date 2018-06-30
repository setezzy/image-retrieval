from __future__ import division
import numpy as np
import pandas as pd
from keras.models import Model
from collections import defaultdict


def get_outputs(input_data, model):
    layer_names = [layer.name for layer in model.layers if
                   'flatten' not in layer.name and 'input' not in layer.name
                   and 'predictions' not in layer.name]
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=[model.get_layer(layer_name).output for layer_name in layer_names])
    layer_outputs = intermediate_layer_model.predict(input_data)
    return layer_outputs, layer_names


def retrieval(query_data, test_data, query_label, test_labels, model):
    '''
    retrieval task
    :param query_data: query input
    :param test_data: test data set
    :param query_label: predicted label of query input
    :param test_labels: test label
    :param model: investigated model
    :return:
        df: dataframe
        f1: F1 measure
    '''
    related = 0
    retrieved_related = 0
    retrieved = 0
    img_index = []
    img_label = []
    similarity = []
    query_outputs = get_outputs(query_data, model)[0]
    query_feat = query_outputs[5][0]

    layer_outputs = get_outputs(test_data, model)[0]
    layer_output = layer_outputs[5]

    for i in xrange(10000):
        test_feat = layer_output[i]
        test_label = np.argmax(test_labels[i])
        if query_label == test_label:
            related += 1
        # use cosine distance as similarity metric
        cos = np.dot(query_feat, test_feat) / (np.linalg.norm(query_feat)*np.linalg.norm(test_feat))
        sim = 0.5 + 0.5*cos
        if sim >= 0.85:
            retrieved += 1
            img_index.append(i)
            similarity.append(sim)
            img_label.append(test_label)
            if query_label == test_label:
                retrieved_related += 1

    # evaluation metric
    recall = retrieved_related / related
    precision = retrieved_related / retrieved
    f1 = recall * precision * 2 / (recall + precision)

    df = pd.DataFrame({'retrieved_index': img_index, 'similarity': similarity, 'label': img_label})
    df = df.sort_values('similarity', ascending=True)
    return df, f1

