# DLforCBIR
Deep Learning for content-based image retrieval with TensorFlow and Keras

## Image preprocessing

**tf.img_trans.py:**

skimage and keras are required in this file. The following transformations are included:

- adjust brightness
- shit, rotate, flip, zoom
- dilation, erosion
- add oblique line
- add salt noise


## LeNet-5
LeNet-5 is a classical CNN model proposed by Yann LeCun. 

See [Gradient-based learning applied to document recognition](https://ieeexplore.ieee.org/abstract/document/726791/) for more details.

**tf.lenet5_model.py:**

LeNet-5 implementation. ReLu is used as the activate function for convlayer and fclayer. And a drop-out layer is added before softmax in this implementation.

**tf.lenet_train.py:**

Train lenet-5 on MNIST.

**tf.retrieval.py:**

Extract features of query image and all retrieval images from fully-connected layer.

Performe feature similarity computation for retrieval task. Cosine similarity is adopted in this implementation.

**tf.retrieval_.py:**

Example of using images stored on disk as tf inputs.

**usage: python retrieval.py -h**
