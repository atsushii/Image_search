import tensorflow as tf

from utils.model import *

class BuildModel(tf.keras.Model):

    def __init__(self, dropout, image_size, number_of_classes=10):
        """
        Define CNN model

        :param dropout: dropout rate
        :param image_size: tuple, (height, width
        :param number_of_classes: integer, number of classes
        """

        super(BuildModel, self).__init__()

        self.batch_normalize_layer = tf.keras.layers.BatchNormalization()

        self.conv_1 = Conv(number_of_filter=64,
                           kernel_size=(3, 3),
                           stride=(1, 1),
                           padding="SAME",
                           activation=tf.nn.relu,
                           max_pool=True,
                           batch_norm=True)

        self.conv_2 = Conv(number_of_filter=128,
                           kernel_size=(3, 3),
                           stride=(1, 1),
                           padding="SAME",
                           activation=tf.nn.relu,
                           max_pool=True,
                           batch_norm=True)

        self.conv_3 = Conv(number_of_filter=256,
                           kernel_size=(5, 5),
                           stride=(1, 1),
                           padding="SAME",
                           activation=tf.nn.relu,
                           max_pool=True,
                           batch_norm=True)

        self.conv_4 = Conv(number_of_filter=512,
                           kernel_size=(5, 5),
                           stride=(1, 1),
                           padding="SAME",
                           activation=tf.nn.relu,
                           max_pool=True,
                           batch_norm=True)

        self.flatten_layer = tf.keras.layers.Flatten()

        self.dense_1 = Dense(units=128,
                             activation=tf.nn.relu,
                             dropout=dropout,
                             batch_norm=True)

        self.dense_2 = Dense(units=256,
                             activation=tf.nn.relu,
                             dropout=dropout,
                             batch_norm=True)

        self.dense_3 = Dense(units=512,
                             activation=tf.nn.relu,
                             dropout=dropout,
                             batch_norm=True)

        self.dense_4 = Dense(units=1024,
                             activation=tf.nn.relu,
                             dropout=dropout,
                             batch_norm=True)

        self.final_dense = tf.keras.layers.Dense(units=number_of_calsses,
                                                 activation=None)

        self.final_softmax = tf.keras.layers.Softmax()

    def call(self, inputs, training):
        x = self.batch_normalize_layer(inputs, training)
        x, conv1 = self.conv_1(x, training)
        x, conv2 = self.conv_2(x, training)
        x, conv3 = self.conv_3(x, training)
        x, conv4 = self.conv_4(x, training)

        x = self.flatten_layer(x)

        x, dense1 = self.dense_1(x, training)
        x, dense2 = self.dense_2(x, training)
        x, dense3 = self.dense_3(x, training)
        x, dense4 = self.dense_4(x, training)

        x = self.final_dense(x)
        output = self.final_softmax(x)

        return output, dense2, dense4
