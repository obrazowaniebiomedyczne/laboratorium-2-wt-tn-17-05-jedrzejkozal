import numpy as np
import cv2


def default_kernel():
    return np.array([[0, 1, 0],
                     [1, 1, 1],
                     [0, 1, 0]], np.uint8)

def bigger_kernel():
    return np.array([[0, 1, 1, 1, 0],
                    [0, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1],
                    [0, 1, 1, 1, 0],
                    [0, 1, 1, 1, 0]], np.uint8)

def quadratic_kernel():
    return np.array([[1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1]], np.uint8)

def open_operation(img, kernel):
    erosion = cv2.erode(img, kernel, iterations = 1)
    dilation = cv2.dilate(erosion, kernel, iterations = 1)

    return dilation

def close_operation(img, kernel):
    dilation = cv2.dilate(img, kernel, iterations = 1)
    erosion = cv2.erode(dilation, kernel, iterations = 1)

    return erosion

# Zadanie na ocenę dostateczną
def renew_pictures():

    number_of_images = 4

    for i in range(1, number_of_images+1):
        if i == 1:
            img = cv2.imread('figures/crushed.png')
        else:
            img = cv2.imread('figures/crushed' + str(i) + '.png')

        if i > 2:
            kernel = quadratic_kernel()
        else:
            kernel = default_kernel()

        img = close_operation(img, kernel)
        img = open_operation(img, kernel)

        cv2.imwrite("results/renewed" + str(i) + ".png", img)


def check_boundires(i_begin, i_end, j_begin, j_end, kernel_shape, img_shape):
    i_begin_mask, i_end_mask, j_begin_mask, j_end_mask = 0, kernel_shape[0], 0, kernel_shape[1]

    if i_begin < 0:
        i_begin_mask = abs(i_begin)
        i_begin = 0
    if i_end > img_shape[0]:
        i_end_mask = i_end - img_shape[0]
        i_end = img_shape[0]
    if j_begin < 0:
        j_begin_mask = abs(j_begin)
        j_begin = 0
    if j_end > img_shape[1]:
        j_end = j_end - img_shape[1]
        j_end = img_shape[1]
    return i_begin, i_end, j_begin, j_end, i_begin_mask, i_end_mask, j_begin_mask, j_end_mask


def get_img_under_mask(img, kernel_shape, i, j):
    kernel_i_half = int(kernel_shape[0]/2)
    kernel_j_half = int(kernel_shape[1]/2)

    i_begin = i - kernel_i_half
    j_begin = j - kernel_j_half

    i_end = i + kernel_i_half + 1
    j_end = j + kernel_j_half + 1

    i_begin, i_end, j_begin, j_end, i_begin_mask, i_end_mask, j_begin_mask, j_end_mask = check_boundires(i_begin, i_end, j_begin, j_end, kernel_shape, img.shape)

    img_under_mask = np.ones(img.shape, dtype=img.dtype)
    img_under_mask = np.multiply(img_under_mask, 255)

    for i_mask, i_img in zip(range(i_begin_mask, i_end_mask), range(i_begin, i_end)):
        for j_mask, j_img in zip(range(j_begin_mask, j_end_mask), range(j_begin, j_end)):
            img_under_mask[i_mask, j_mask] = img[i_img, j_img]

    return img_under_mask


def implication(p, q):
    return (not p) or q


def apply_mask(kernel, img_under_kernel):
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            if not implication(kernel[i,j], img_under_kernel[i,j]):
                return True
    return False


def erosion(img, kernel):
    new_image = np.full(img.shape, 255, dtype=img.dtype)

    for k in range(img.shape[2]):
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if apply_mask(kernel, get_img_under_mask(img[:,:,k], kernel.shape, i, j)):
                    new_image[i,j,k] = 0

    return new_image


# Zadanie na ocenę dobrą
def own_simple_erosion(image):
    kernel = np.zeros((3,3),np.uint8)
    kernel[:,1] = 1
    kernel[1,:] = 1

    return erosion(image, kernel)


# Zadanie na ocenę bardzo dobrą
def own_erosion(image, kernel=None):
    new_image = np.zeros(image.shape, dtype=image.dtype)
    if kernel is None:
        kernel = np.array([[0, 1, 0],
                           [1, 1, 1],
                           [0, 1, 0]])

    return erosion(image, kernel)
