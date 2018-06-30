'''
image transformation
'''

import numpy as np
import random
from skimage import draw
from skimage import morphology as mp
from skimage import exposure as ep
from keras.preprocessing import image


def deprocess_img(x):
    # convert tensor to a valid image
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x.reshape(x.shape[1], x.shape[2])


def light(img):
    # adjust_gamma(x, gamma)
    # gamma > 1 --- darker
    # gamma < 1 --- brighter
    img = img.astype(np.float32)
    return ep.adjust_gamma(img, random.uniform(0.7, 1.05))


def shift(img):
    # random_shift(x, wrg, hrg, row_axis=1, col_axis=2, channel_axis=0, fill_mode='nearest', cval=0.0)
    img = img.reshape(1, 28, 28)
    wrg = 0.2
    hrg = 0.2
    return image.random_shift(img, wrg, hrg).reshape(28, 28)


def rotate(img):
    # random_rotation(x, rg, row_axis=1, col_axis=2, channel_axis=0, fill_mode='nearest', cval=0.0)
    img = img.reshape(1, 28, 28)
    rg = 30
    return image.random_rotation(img, rg).reshape(28, 28)


def flip(img):
    # flip_axis(x, axis)
    # horizontal
    return image.flip_axis(img, 1)


def zoom(img):
    # random_zoom(x, zoom_range, row_axis=1, col_axis=2, channel_axis=0, fill_mode='nearest', cval=0.0)
    img = img.reshape(1, 28, 28)
    zoom_range = (0.6, 1.3)
    return image.random_zoom(img, zoom_range).reshape(28, 28)


def dilation(img):
    # return greyscale morphological dilation of an image
    return mp.dilation(img, mp.square(2, dtype=np.uint8))


def erosion(img):
    # return greyscale morphological erosion of an image
    return mp.erosion(img, mp.square(2, dtype=np.uint8))


def draw_line(img):
    img = img.reshape(28, 28)
    # draw a straight line across the digit
    r0 = random.randint(1, 27)
    c0 = random.randint(1, 4)
    r1 = random.randint(1, 27)
    c1 = random.randint(24, 27)
    rr, cc = draw.line(r0, c0, r1, c1)
    img[rr, cc] = 255
    return img


def s_p_noise(img):
    # salt and pepper noise
    img = img.reshape(28, 28)
    rows, cols = img.shape
    for i in range(50):
        x = np.random.randint(0, rows)
        y = np.random.randint(0, cols)
        img[x, y] = 255
        x_ = np.random.randint(4, 24)
        y_ = np.random.randint(4, 24)
        img[x_, y_] = 0
    return img




