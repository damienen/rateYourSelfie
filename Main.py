import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models, layers

import utils.Data_Loading as data

male_train_data, female_train_data, male_labels, female_labels = data.load_data()

male_train_data, female_train_data = male_train_data / 255, female_train_data / 255

input_shape = (350, 350, 3)

model = models.Sequential()

model.add(layers.Conv2D(
    filters=96,
    kernel_size=13,
    strides=4,
    input_shape=input_shape,
    activation='relu'
))

model.add(layers.MaxPool2D(
    pool_size=3,
    strides=2
))

model.add(layers.Conv2D(
    filters=256,
    kernel_size=5,
    padding='same',
    activation='relu'
))

model.add(layers.MaxPool2D(
    pool_size=3,
    strides=2,
))

model.add(layers.Conv2D(
    filters=384,
    kernel_size=3,
    padding='same',
    activation='relu'
))

model.add(layers.MaxPool2D(
    pool_size=3,
    strides=2,
))

model.add(layers.Flatten())

model.add(layers.Dense(
    units=1024,
    activation='relu',
    kernel_initializer=tf.keras.initializers.GlorotUniform()
))

model.add(layers.Dense(
    units=128,
    activation='relu',
    kernel_initializer=tf.keras.initializers.GlorotUniform()
))

model.add(layers.Dense(
    units=1,
    kernel_initializer=tf.keras.initializers.GlorotUniform()
))

print(model.summary())

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False),
              loss=tf.keras.losses.logcosh,
              metrics=['mae', 'mse']
              )

history = model.fit(
    male_train_data,
    male_labels,
    epochs=15,
    validation_split=0.06,
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)]
)

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

test_loss, test_mae, test_mse = model.evaluate(male_train_data[700:750], male_labels[700:750], verbose=1)

print(test_loss)
print(test_mae)
print(test_mse)

# serialize model to JSON
model_json = model.to_json()
with open("saved_model/model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("saved_model/model.h5")
print("Saved model to disk")
