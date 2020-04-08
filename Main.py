import tensorflow as tf
import numpy as np
import Convolution.functions as func
import utils.Data_Loading as data

import matplotlib.pyplot as plt

male_train_data, female_train_data, male_test_data, female_test_data, male_train_labels, female_train_labels, male_test_labels, female_test_labels = data.load_data()

input_shape = (350, 350, 3)
conv_filter_shape = [(11, 96), (5, 256), (3, 384)]
conv_stride = [4, 1, 1]
pool_size = [3, 3, 3]
pool_stride = [2, 2, 2]

nr_nodes = [1024, 64, 1]

model = func.forward_convolution(input_shape, conv_filter_shape, conv_stride, pool_size, pool_stride)

model = func.forward_fully_connected(model, nr_nodes)

print(model.summary())

model.compile(optimizer='adam',
              loss=tf.keras.losses.MSE,
              metrics=['mae', 'mse']
              )

history = model.fit(male_train_data, male_train_labels, epochs=10, validation_data=(male_test_data, male_test_labels))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])

plt.title('model mae')
plt.ylabel('mae')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['mse'])
plt.plot(history.history['val_mse'])

plt.title('model mse')
plt.ylabel('mse')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

test_loss, test_mae, test_mse = model.evaluate(male_test_data, male_test_labels, verbose=1)

print(test_loss)
print(test_mae)
print(test_mse)

result = model.predict(male_test_data[:10])

print(result)
print(male_test_labels[:10])
