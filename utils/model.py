import tensorflow as tf


class Conv(tf.keras.layers.Layer):

    def __init__(self, number_of_filter, kernel_size, stride=(1, 1),
                 padding="SAME", activation="tf.nn.relu",
                 max_pool=True, batch_norm=True):

        """
        define convolutional block layer

        :param number_of_filter: interger, number of filter
        :param kernel_size: tuple, size of conv layer kernel
        :param padding: String, type of padding SAME or VALID
        :param activation: tf.object, activation functuin used on the layer
        :param max_pool: boolean, true conv layer use max pooling
        :param batch_norm: boolean, true conv layer use batch normalization
        """

        super(Conv, self).__init__()

        self.conv_layer = tf.keras.layers.Conv2D(filters=number_of_filter,
                                                 kernel_size=kernel_size,
                                                 strides=stride,
                                                 padding=padding,
                                                 activation=activation)

        self.max_pool = max_pool
        if max_pool:
            self.max_pool_layer = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
                                                               strides=(2, 2),
                                                               padding="SAME")

        self.batch_norm = batch_norm
        if batch_norm:
            self.batch_norm_layer = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training):
        conv_features = x = self.conv_layer(inputs)
        if self.max_pool:
            x = self.max_pool_layer(x)
        if self.batch_norm:
            x = self.batch_norm_layer(x, training)

        return x, conv_features

class Dense(tf.keras.layers.Layer):
    def __init__(self, units, activation=tf.nn.relu, dropout=None, batch_norm=True):
        """
        define Dense layer

        :param units: interger, number of neurons
        :param activation: tf.object, activation functuin used on the layer
        :param dropout: dropout rate
        :param batch_norm: boolean, true conv layer use batch normalization
        """

        super(Dense, self).__init__()

        self.dense_layer = tf.keras.layers.Dense(units, activation=activation)

        self.dropout = dropout
        if dropout is not None:
            self.dropout_layer = tf.keras.layers.Dropout(dropout)

        self.batch_norm = batch_norm
        if batch_norm:
            self.batch_norm_layer = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training):
        dense_feature = x = self.dense_layer(inputs)
        if self.dropout is not None:
            x = self.dropout_layer(x, training)
        if self.batch_norm:
            x = self.batch_norm_layer(x, training)
        return x, dense_feature