import numpy as np
import os
from scipy import stats

# TensorFlow
import tensorflow as tf
from keras import activations
 
print(tf.__version__)

def circulo(num_datos=100, R=1, minimo=0, maximo=1, latitud=0, longitud=0):
    pi = np.pi

    r = R * np.sqrt(stats.truncnorm.rvs(minimo, maximo, size=num_datos)) * 10
    theta = stats.truncnorm.rvs(minimo, maximo, size=num_datos) * 2 * pi * 10

    x = np.cos(theta) * r
    y = np.sin(theta) * r

    x = np.round(x + longitud, 3)
    y = np.round(y + latitud, 3)

    df = np.column_stack([x, y])
    return df

N = 250

datos_newyork = circulo(num_datos=N, R=1.5, latitud=40.4526, longitud=-73.5816)
datos_auckland = circulo(num_datos=N, R=1, latitud= -36.8484, longitud= 174.7633)
X = np.concatenate([datos_newyork, datos_auckland])
X = np.round(X, 3)
print ('X : ', X)

y = [0] * N + [1] * N
y = np.array(y).reshape(len(y), 1)
print ('y : ', y)

train_end = int(0.6 * len(X))
#print (train_end)
test_start = int(0.8 * len(X))
#print (test_start)
X_train, y_train = X[:train_end], y[:train_end]
X_test, y_test = X[test_start:], y[test_start:]
X_val, y_val = X[train_end:test_start], y[train_end:test_start]

tf.keras.backend.clear_session()
linear_model = tf.keras.models.Sequential([tf.keras.layers.Dense(units=4, input_shape=[2], activation=activations.relu, name='relu1'),
                                           tf.keras.layers.Dense(units=8, activation=activations.relu, name='relu2'),
                                           tf.keras.layers.Dense(units=1, activation=activations.sigmoid, name='sigmoid')])
linear_model.compile(optimizer=tf.keras.optimizers.SGD(), loss=tf.keras.losses.MeanSquaredError)
print(linear_model.summary())

linear_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=500)
w = linear_model.layers[0].get_weights()[0]
b = linear_model.layers[0].get_weights()[1]
print('W 0', w)
print('b 0', b)
w = linear_model.layers[1].get_weights()[0]
b = linear_model.layers[1].get_weights()[1]
print('W 1', w)
print('b 1', b)
w = linear_model.layers[2].get_weights()[0]
b = linear_model.layers[2].get_weights()[1]
print('W 2', w)
print('b 2', b)

print('predict city 1 : new york')
newyork_matrix = tf.constant([ [40.7580, -73.9855],
                           [40.7851, -73.9683 ], 
                           [40.6892, -74.0445] ], tf.float32)

#print(linear_model.predict([[-43.598 -28.107][-46.268 -14.62 ] [-45.154 -3.249] [-46.52 -21.315][-41.719 -10.532][-48.291 -28.376]] ))   
print(linear_model.predict(newyork_matrix).tolist() )   
print('predict city 2 : auckland')
auckland_matrix = tf.constant([ [-36.8485, 174.7633],
                           [-36.8527, 174.7699] ], tf.float32)

print(linear_model.predict(auckland_matrix).tolist() ) 
#print(linear_model.predict([[-43.598 -28.107],[-46.268 -14.62]] ).tolist() )   

# export_path = 'linear-model/1/'
# tf.saved_model.save(linear_model, os.path.join('./',export_path))