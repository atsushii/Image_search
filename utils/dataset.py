import cv2
import os
import numpy as np
import pickle


def image_load(path, image_size):
    """
    Load image

    :param path: String, path to image
    :image_size: tuple, size of output image
    """

    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, image_size, cv2.INTER_CUBIC)
    return image


def data_preprocessing(data_path, labels_path, image_size, image_path_pickle):
    """
    Load image and labels

    :param data_path: String, path to train and test data
    :param labels_path: String, path to label
    :param image_size: tuple, single imaze size
    :param image_path_pickle: String, name of a pickle file where all image will be saved
    """
    with open(labels_path) as f:
        classes = f.read().split("\n")[: -1]

    images = []
    labels = []
    # path to pickle
    image_paths = []

    for image_name in os.listdir(data_path):
        try:

            # create full path to train data
            image_path = os.path.join(data_path, image_name)
            images.append(image_load(image_path, image_size))
            image_paths.append(image_path)
            for idx in range(len(classes)):
                if classes[idx] in image_name:
                    labels.append(idx)
        except:
            pass

    with open(image_path_pickle + ".pickle", "wb") as f:
        pickle.dump(image_paths, f)

    assert len(images) == len(labels)
    return np.array(images, dtype="float32"), np.array(labels, dtype="float32")
