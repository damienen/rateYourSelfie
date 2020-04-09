import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.models import model_from_json

import numpy as np
from matplotlib import image
import matplotlib.pyplot as plt

json_file = open('saved_model/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("saved_model/model.h5")
print("Loaded model from disk")

loaded_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False),
                     loss=tf.keras.losses.logcosh,
                     metrics=['mae', 'mse']
                     )


imgs = np.zeros((2, 350, 350, 3))
imgs[0] = np.array(image.imread("Images/Damian.jpg"))
imgs[1] = np.array(image.imread("Images/Danut.jpg"))

imgs = imgs/255


print(imgs.shape)


pred = loaded_model.predict(imgs)
print(pred)
