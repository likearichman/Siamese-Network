import numpy as np
import random

import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, Lambda, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt

import datetime

#Group Member: Yuchen Jiang (N9573950)(Submitter)
#               Jin Zhao (N9800174)
#               Jinning Guo (N9858598)
data_set = fashion_mnist



# Calculat distance


def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

# distance shape


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

# loss fuction


def contrastive_loss(y_true, y_pred):
    margin = 1
    sqaure_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)


'''
    the data from X is the corresponding image data set
    class_indices corresponds to the index of the label in the train set
'''


def create_pairs(x, class_indices):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    n = min([len(class_indices[d]) for d in range(len(class_indices))]) - 1
    for d in range(len(class_indices)):
        for i in range(n):
            z1, z2 = class_indices[d][i], class_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, len(class_indices))
            dn = (d + inc) % len(class_indices)
            z1, z2 = class_indices[d][i], class_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1.0, 0.0]
    return np.array(pairs), np.array(labels)

# Random data


def shuffle(x, y):
    # indices = the number of images in the source data set
    index = np.arange(len(y))
    np.random.shuffle(index)
    return x[index], y[index]

# Building a model


def create_base_network(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input = Input(shape=input_shape)
    x = Conv2D(16, kernel_size=(3, 3),
               activation='relu', kernel_regularizer=regularizers.l2(0.06), padding='same')(input)
    # kernel_regularizer=regularizers.l2(0.06),
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(64, kernel_size=(3, 3), activation='relu',
               kernel_regularizer=regularizers.l2(0.1))(x)
    # ,kernel_regularizer=regularizers.l2(0.1)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(32, kernel_size=(3, 3), activation='relu',
               kernel_regularizer=regularizers.l2(0.08))(x)
    # ,kernel_regularizer = regularizers.l2(0.08)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    return Model(input, x)

# calcuate accuracy


def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


def plot_train_history(history, train_metrics, val_metrics):
    plt.plot(history.history.get(train_metrics), '-o')
    plt.plot(history.history.get(val_metrics), '-o')
    plt.ylabel(train_metrics)
    plt.xlabel('Epochs')
    plt.legend(['train', 'validation'])


def showImages(x, y):
    plt.figure(figsize=(10, 10))
    for i in range(25):
        for j in range(2):
            plt.subplot(8, 8, i*2+j+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(x[i][j].reshape(
                28, 28), cmap=plt.cm.binary)
        plt.subplot(8, 8, i*2+1)
        plt.xlabel(['true', 'false'][y[i]])
    plt.show()


# train_images, test_images: uint8 Grayscale image，size (num_samples, 28, 28)。
# train_labels, test_labels: uint8 Digital label（Range from 0 to 9），size (num_samples,)


# Category	Description      
# 0	        T-shirt/top	 
# 1	        Trouser	     
# 2	        Pullover	 
# 3	        Dress	    
# 4	        Coat	     
# 5	        Sandal	     
# 6	        Shirt	     
# 7	        Sneaker	     
# 8	        Bag	         
# 9	        Ankle boot	 
class_names = ['top', 'trouser', 'pullover', 'dress', 'coat',
               'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']

# num_classes = 10
# Tag training type
train_classes = [0, 1, 2, 4, 5, 9]
# Tag test type
test_classes = [3, 7, 8, 6]

# input image dimensions
img_rows, img_cols = 28, 28

epochs = 3
#epochs = 50
# load data
(train_images, train_labels), (test_images, test_labels) = data_set.load_data()
# concat train and test data
train_images = np.concatenate((train_images, test_images))
train_labels = np.concatenate((train_labels, test_labels))


if K.image_data_format() == 'channels_first':
    train_images = train_images.reshape(
        train_images.shape[0], 1, img_rows, img_cols)
    test_images = test_images.reshape(
        test_images.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    train_images = train_images.reshape(
        train_images.shape[0], img_rows, img_cols, 1)
    test_images = test_images.reshape(
        test_images.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# Normalization from 0 - 255 to 0 - 1
train_images, test_images = train_images.astype(
    'float32'), test_images.astype('float32')
train_images, test_images = train_images / 255.0, test_images / 255.0

input_shape = train_images.shape[1:]

# create training+test positive and negative pairs

class_indices = [np.where(train_labels == i)[0] for i in train_classes]


# class_count = [len(np.where(train_labels == i)[0]) for i in range(10)]

# Match the corresponding test image data in pairs using the trainings of the training data set obtained above
pairs, y = create_pairs(train_images, class_indices)
pairs_for_012459, label_for_012459 = pairs, y
# Take 80% of the data set to train another 20% as a test set
train_y_len = round(len(y)*0.8)

# Random data set，if not，The last few types will not be included in the calculation
shuffled_pairs, shuffled_y = shuffle(pairs, y)
train_pairs, train_y = shuffled_pairs[0:train_y_len], shuffled_y[0:train_y_len]

# use the 20% data set as the training set
test_pairs, test_y = shuffled_pairs[train_y_len:], shuffled_y[train_y_len:]


class_indices = [np.where(train_labels == i)[0] for i in test_classes]
ftest_pairs, ftest_y = create_pairs(train_images, class_indices)

# showImages(train_pairs, train_y)

# network definition
base_network = create_base_network(input_shape)

input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)

# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
processed_a = base_network(input_a)
processed_b = base_network(input_b)

distance = Lambda(euclidean_distance,
                  output_shape=eucl_dist_output_shape)([processed_a, processed_b])

model = Model([input_a, input_b], distance)

# generate tensorboard flowchart
log_dir = "logs/fit/"  # + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=1)

# trainRMSprop
rms = RMSprop()
# Compile model
model.compile(loss=contrastive_loss, optimizer=rms, metrics=[accuracy])
# train model
history = model.fit([train_pairs[:, 0], train_pairs[:, 1]], train_y, batch_size=128,
                    epochs=epochs, validation_data=([test_pairs[:, 0], test_pairs[:, 1]], test_y), callbacks=[tensorboard_callback])


# compute final accuracy on training and test sets
y_pred = model.predict([train_pairs[:, 0], train_pairs[:, 1]])
tr_acc = compute_accuracy(train_y, y_pred)
y_pred = model.predict([test_pairs[:, 0], test_pairs[:, 1]])
te_acc = compute_accuracy(test_y, y_pred)

# compute_accuracy(ftest_y, model.predict([ftest_pairs[:, 0], ftest_pairs[:, 1]]))
print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))

y_pred = model.predict([pairs_for_012459[:, 0], pairs_for_012459[:, 1]])
tr_acc = compute_accuracy(label_for_012459, y_pred)
print('* Accuracy on 012459 set: %0.2f%%' % (100 * tr_acc))

class_indices = [np.where(train_labels == i)[0] for i in range(10)]
all_pairs, all_label = create_pairs(train_images, class_indices)
y_pred = model.predict([all_pairs[:, 0], all_pairs[:, 1]])
tr_acc = compute_accuracy(all_label, y_pred)
print('* Accuracy on all set: %0.2f%%' % (100 * tr_acc))

y_pred = model.predict([ftest_pairs[:, 0], ftest_pairs[:, 1]])
tx_acc = compute_accuracy(ftest_y, y_pred)
print('* Accuracy on final_test set: %0.2f%%' % (100 * tx_acc))

