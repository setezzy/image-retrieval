# image-retrieval
Deep Learning for content-based image retrieval with TensorFlow and Keras


## tensorflow version

**usage: python retrieval.py -h**

**img_trans.py:**

skimage and keras are required in this file. The following transformations are included:

- adjust brightness
- shit, rotate, flip, zoom
- dilation, erosion
- add oblique line
- add salt noise

**lenet5_model.py:**

LeNet-5 is a classical CNN model proposed by Yann LeCun. 

See [Gradient-based learning applied to document recognition](https://ieeexplore.ieee.org/abstract/document/726791/) for more details.

ReLu is used as the activate function for convlayer and fclayer. Drop-out layer is added before softmax in this implementation.

**lenet_train.py:**

Train lenet-5 on MNIST.

**retrieval.py:**

Extract features of query image and all retrieval images from fully-connected layer.

Perform feature similarity computation for retrieval task. Cosine similarity is adopted in this implementation.

**retrieval_.py:**

Example of using images stored on disk as tf inputs.


## keras version

**utils.py:**

Image preprocessing.

**funcs.py:**

Image retrieval function.

**main.py:**

Main function.
