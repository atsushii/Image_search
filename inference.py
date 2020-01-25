import cv2

from utils.utils import *
from utils.dataset import image_load


def inference(model, train_set_vector, upload_image_path, image_size, distance="hamming"):
    """
    doing simple inference for single upload image

    :param model: CNN model
    :param train_set_vector: load train set vector
    :param upload_image_path: String, path to upload image
    :param image_size: tuple, (height, width)
    :param distance: String, type of distance to used
    """

    image = image_load(upload_image_path, image_size)
    channel = cv2.split(image)
    feature = []

    for cha in channel:
        hist = cv2.calcHist([cha], [0], None, [256], [0, 256])
        feature.append(hist)

    color_featuer = np.vstack(feature).T

    _, dense_2_features, dense_4_features = model(image, training=False)

    closest_id = None

    if distance == "hamming":
        dense_2_features = np.where(dense_2_features < 0.5, 0, 1)
        dense_4_features = np.where(dense_4_features < 0.5, 0, 1)

        upload_image_vector = np.hstack((dense_2_features, dense_4_features))
        closest_id = hamming_distance(train_set_vector, upload_image_vector)
    elif distance == "cosine":
        upload_image_vector = np.hstack(train_set_vector, upload_image_vector)
        closest_id = cosine_distance(train_set_vector, upload_image_vector)
    return closest_id

def inference_with_color_filter(model, train_set_vector, upload_image_path, color_vector, image_size,
                                distance="hamming"):
    """
    doing simple inference for single image

    :param model: CNN model
    :param train_set_vector: loaded train set vector
    :param upload_image_path: String, path to the load image
    :param color_vector: train set vector feature vector
    :param image_size: tuple, (height, width)
    :param distance: string, type of distance
    """

    image = image_load(upload_image_path, image_size)

    # calculate color histgram of the query image
    channel = cv2.split(image)
    feature = []

    for chan in channel:
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        feature.append(hist)

    color_feature = np.vstack(feature).T

    image_input = np.expand_dims(image, axis=0).astype("float32")
    _, dense_2_feature, dense_4_feature = model(image_input, training=False)

    closest_ids = None
    if distance == "hamming":

        dense_2_feature = np.where(dense_2_feature < 0.5, 0, 1)
        dense_4_feature = np.where(dense_4_feature < 0.5, 0, 1)

        upload_image_vector = np.hstack((dense_2_feature, dense_4_feature))
        closest_id = hamming_distance(train_set_vector, upload_image_vector)

        # compare color feature between query image and closest image
        closest_id = compare_color(np.array(color_vector)[closest_id], color_feature, closest_id)

    elif distance == "cosine":

        upload_image_vector = np.hstack((dense_2_feature, dense_4_feature))
        closest_id = cosine_distance(train_set_vector, upload_image_vector)

        # compare color feature between query image and closest image
        closest_id = compare_color(np.array(color_vector)[closest_id], color_feature, closest_id)
    return closest_id