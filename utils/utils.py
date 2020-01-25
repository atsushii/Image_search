import numpy as np
from scipy.spatial.distance import hamming, cosine, euclidean


def cosine_distance(training_set_vector, query_vector, top_n=50):
    """
    calculate cosine distance between query image and all trianing set image

    :param training_set_vector: numpy Matrix, vectors for all images in training set
    :param query_vector: numpy vector, query image(new image)
    :param top_n: interger, number of closest image to return
    """

    distance = []

    for i in range(len(training_set_vector)):  # for train dataset 50k image
        distance.append(cosine(training_set_vector[i], query_vector[0]))

    return np.argsort(distance)[:top_n]


def hamming_distance(training_set_vector, query_vector, top_n=50):
    """
    calculate haming distance between query image and all training images

    :param training_set_vector: numpy Matrix, vectors for all images in training set
    :param query_vector: numpy vector, query image(new image)
    :param top_n: interger, number of closest image to return
    """

    distance = []

    for i in range(len(training_set_vector)):
        distance.append(hamming(training_set_vector[i], query_vector[0]))

    return np.argsort(distance)[:top_n]


def cal_accuracy(y_true, y_pred):
    """
    calculate accuracy of model on softmax outputs

    :param y_true: numpy array, true labels of each sample
    :param y_pred: numpy matrix, softmax probabilities
    """

    assert len(y_true) == len(y_pred)

    correct = 0

    for i in range(len(y_true)):
        if np.argmax(y_pred[i]) == y_true[i]:
            correct += 1

    return correct / len(y_true)


def compare_color(color_vector, upload_image, ids):
    """
    Compare color vector of closest image from train set with a color vector of a upload image

    :param color_vector: color feature vector of closest train set image to the upload image
    :param upload_image: color vector of upload image
    param ids: indices of training image being closest to the upload image (output from a distance func)
    """

    color_distance = []

    for i in (range(len(color_vector))):
        color_distance.append(euclidean(color_vector[i], upload_image))

    return ids[np.argsort(color_distance)[:10]]