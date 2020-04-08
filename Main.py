import tensorflow as tf
from tensorflow.keras import datasets
import Convolution.functions as func

import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

train_images, test_images = train_images / 255.0, test_images / 255.0

input_shape = (350, 350, 3)
conv_filter_shape = [(11, 96), (5, 256), (3, 384)]
conv_stride = [4, 1, 1]
pool_size = [3, 3, 3]
pool_stride = [2, 2, 2]

nr_nodes = [1024, 64, 64]

model = func.forward_convolution(input_shape, conv_filter_shape, conv_stride, pool_size, pool_stride)

model = func.forward_fully_connected(model, nr_nodes)

print(model.summary())

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy']
              )

history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.ylim([0.5, 1])
plt.legend(loc='lower right')

plt.show()

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print(test_acc)
