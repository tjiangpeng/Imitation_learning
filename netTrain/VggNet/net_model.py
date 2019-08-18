"""VGG16 model for Keras.
# Reference
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](
    https://arxiv.org/abs/1409.1556) (ICLR 2015)
"""
import os
import tensorflow as tf
from tensorflow import keras

WEIGHTS_PATH = ('https://github.com/fchollet/deep-learning-models/'
                'releases/download/v0.1/'
                'vgg16_weights_tf_dim_ordering_tf_kernels.h5')
WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
                       'releases/download/v0.1/'
                       'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')

lrelu = lambda x: tf.keras.activations.relu(x, alpha=0.1)

IMAGE_HEIGHT = 384
IMAGE_WIDTH = 384
NUM_CHANNELS = 3
# NUM_TIME_SEQUENCE = 20
# NUM_HIDDEN_NODES = 2500

def VGG16(include_top=False,
          weights='imagenet',
          input_tensor=None,
          input_shape=None,
          pooling=None,
          activation_fn='relu',
          classes=1000):
    """Instantiates the VGG16 architecture.
    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.
    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor
            (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)`
            (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 input channels,
            and width and height should be no smaller than 32.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional block.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """

    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')

    if input_tensor is None:
        img_input = keras.layers.Input(shape=input_shape)
    else:
        img_input = keras.layers.Input(tensor=input_tensor, shape=input_shape)

    # Block 1
    x = keras.layers.Conv2D(64, (3, 3),
                            activation=activation_fn,
                            padding='same',
                            name='block1_conv1')(img_input)
    x = keras.layers.Conv2D(64, (3, 3),
                            activation=activation_fn,
                            padding='same',
                            name='block1_conv2')(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = keras.layers.Conv2D(128, (3, 3),
                            activation=activation_fn,
                            padding='same',
                            name='block2_conv1')(x)
    x = keras.layers.Conv2D(128, (3, 3),
                            activation=activation_fn,
                            padding='same',
                            name='block2_conv2')(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = keras.layers.Conv2D(256, (3, 3),
                            activation=activation_fn,
                            padding='same',
                            name='block3_conv1')(x)
    x = keras.layers.Conv2D(256, (3, 3),
                            activation=activation_fn,
                            padding='same',
                            name='block3_conv2')(x)
    x = keras.layers.Conv2D(256, (3, 3),
                            activation=activation_fn,
                            padding='same',
                            name='block3_conv3')(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = keras.layers.Conv2D(512, (3, 3),
                            activation=activation_fn,
                            padding='same',
                            name='block4_conv1')(x)
    x = keras.layers.Conv2D(512, (3, 3),
                            activation=activation_fn,
                            padding='same',
                            name='block4_conv2')(x)
    x = keras.layers.Conv2D(512, (3, 3),
                            activation=activation_fn,
                            padding='same',
                            name='block4_conv3')(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = keras.layers.Conv2D(512, (3, 3),
                            activation=activation_fn,
                            padding='same',
                            name='block5_conv1')(x)
    x = keras.layers.Conv2D(512, (3, 3),
                            activation=activation_fn,
                            padding='same',
                            name='block5_conv2')(x)
    x = keras.layers.Conv2D(512, (3, 3),
                            activation=activation_fn,
                            padding='same',
                            name='block5_conv3')(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    if include_top:
        # Classification block
        x = keras.layers.Flatten(name='flatten')(x)
        x = keras.layers.Dense(4096, activation='relu', name='fc1')(x)
        x = keras.layers.Dense(4096, activation='relu', name='fc2')(x)
        x = keras.layers.Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = keras.layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = keras.layers.GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = keras.utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = tf.keras.Model(inputs, x, name='vgg16')

    # Load weights.
    if weights == 'imagenet':
        if include_top:
            weights_path = keras.utils.get_file(
                'vgg16_weights_tf_dim_ordering_tf_kernels.h5',
                WEIGHTS_PATH,
                cache_subdir='models',
                file_hash='64373286793e3c8b2b4e3219cbf3544b')
        else:
            weights_path = keras.utils.get_file(
                'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                WEIGHTS_PATH_NO_TOP,
                cache_subdir='models',
                file_hash='6d6bbae143d832006294945121d1f1fc')
        model.load_weights(weights_path)

    elif weights is not None:
        model.load_weights(weights)

    return model


def VGG16_FL(num_node=4096, num_time_sequence=20, weight=None):
    vgg_model = VGG16(include_top=False, weights=None,
                      input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS),
                      pooling='max')
    last_layer = vgg_model.get_layer('block5_pool').output
    x = keras.layers.Flatten(name='flatten')(last_layer)
    x = keras.layers.Dense(num_node, activation='linear', name='fc1')(x)
    x = keras.layers.LeakyReLU(alpha=0.1)(x)
    x = keras.layers.Dense(num_node, activation='linear', name='fc2')(x)
    x = keras.layers.LeakyReLU(alpha=0.1)(x)
    out = keras.layers.Dense(num_time_sequence*2, activation='linear', name='fc3')(x)
    out = keras.layers.LeakyReLU(alpha=0.1)(out)
    model = keras.Model(vgg_model.input, out)

    if weight is not None:
        model.load_weights(weight)

    return model
