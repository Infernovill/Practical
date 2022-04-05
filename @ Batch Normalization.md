```python
! pip install extra_keras_datasets
```

    Requirement already satisfied: extra_keras_datasets in c:\users\rafyq\anaconda3\lib\site-packages (1.2.0)
    Requirement already satisfied: scikit-learn in c:\users\rafyq\anaconda3\lib\site-packages (from extra_keras_datasets) (0.24.1)
    Requirement already satisfied: numpy in c:\users\rafyq\anaconda3\lib\site-packages (from extra_keras_datasets) (1.20.1)
    Requirement already satisfied: scipy in c:\users\rafyq\anaconda3\lib\site-packages (from extra_keras_datasets) (1.6.2)
    Requirement already satisfied: pandas in c:\users\rafyq\anaconda3\lib\site-packages (from extra_keras_datasets) (1.2.4)
    Requirement already satisfied: python-dateutil>=2.7.3 in c:\users\rafyq\anaconda3\lib\site-packages (from pandas->extra_keras_datasets) (2.8.1)
    Requirement already satisfied: pytz>=2017.3 in c:\users\rafyq\anaconda3\lib\site-packages (from pandas->extra_keras_datasets) (2021.1)
    Requirement already satisfied: six>=1.5 in c:\users\rafyq\anaconda3\lib\site-packages (from python-dateutil>=2.7.3->pandas->extra_keras_datasets) (1.15.0)
    Requirement already satisfied: threadpoolctl>=2.0.0 in c:\users\rafyq\anaconda3\lib\site-packages (from scikit-learn->extra_keras_datasets) (2.1.0)
    Requirement already satisfied: joblib>=0.11 in c:\users\rafyq\anaconda3\lib\site-packages (from scikit-learn->extra_keras_datasets) (1.0.1)
    


```python
from extra_keras_datasets import kmnist
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
```


```python
# Model configuration
batch_size = 250
no_epochs = 25
no_classes = 10
validation_split = 0.2
verbosity = 1
```


```python
from keras.datasets import mnist
```


```python
# Loading KMNIST dataset
(input_train, target_train), (input_test, target_test) = mnist.load_data()
```


```python
# Shape of the input sets
input_train_shape = input_train.shape
input_test_shape = input_test.shape
```


```python
# Keras layer input shape
input_shape = (input_train_shape[1], input_train_shape[2], 1)
```


```python
# Reshaping the training data to include channels
input_train = input_train.reshape(input_train_shape[0], input_train_shape[1], input_train_shape[2], 1)
input_test = input_test.reshape(input_test_shape[0], input_test_shape[1], input_test_shape[2], 1)
```


```python
# Parsing numbers as floats
input_train = input_train.astype('float32')
input_test = input_test.astype('float32')
```


```python
# Normalizing input data
input_train = input_train / 255
input_test = input_test / 255
```


```python
# Creating the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(no_classes, activation='softmax'))
```


```python
# Compiling the model
model.compile(loss=tensorflow.keras.losses.sparse_categorical_crossentropy,
              optimizer=tensorflow.keras.optimizers.Adam(),
              metrics=['accuracy'])
```


```python
# Fitting data to model
history = model.fit(input_train, target_train,
            batch_size=batch_size,
            epochs=no_epochs,
            verbose=verbosity,
            validation_split=validation_split)
```

    Epoch 1/25
    192/192 [==============================] - 248s 1s/step - loss: 0.1225 - accuracy: 0.9630 - val_loss: 4.7744 - val_accuracy: 0.1060
    Epoch 2/25
    192/192 [==============================] - 248s 1s/step - loss: 0.0281 - accuracy: 0.9922 - val_loss: 1.7732 - val_accuracy: 0.5188
    Epoch 3/25
    192/192 [==============================] - 257s 1s/step - loss: 0.0133 - accuracy: 0.9966 - val_loss: 0.1548 - val_accuracy: 0.9537
    Epoch 4/25
    192/192 [==============================] - 231s 1s/step - loss: 0.0069 - accuracy: 0.9986 - val_loss: 0.0475 - val_accuracy: 0.9868
    Epoch 5/25
    192/192 [==============================] - 201s 1s/step - loss: 0.0032 - accuracy: 0.9996 - val_loss: 0.0349 - val_accuracy: 0.9894
    Epoch 6/25
    192/192 [==============================] - 170s 887ms/step - loss: 0.0013 - accuracy: 0.9999 - val_loss: 0.0341 - val_accuracy: 0.9897
    Epoch 7/25
    192/192 [==============================] - 143s 742ms/step - loss: 6.8903e-04 - accuracy: 1.0000 - val_loss: 0.0296 - val_accuracy: 0.9920
    Epoch 8/25
    192/192 [==============================] - 164s 855ms/step - loss: 3.6649e-04 - accuracy: 1.0000 - val_loss: 0.0301 - val_accuracy: 0.9919
    Epoch 9/25
    192/192 [==============================] - 121s 632ms/step - loss: 2.7514e-04 - accuracy: 1.0000 - val_loss: 0.0298 - val_accuracy: 0.9923
    Epoch 10/25
    192/192 [==============================] - 122s 636ms/step - loss: 2.3270e-04 - accuracy: 1.0000 - val_loss: 0.0304 - val_accuracy: 0.9923
    Epoch 11/25
    192/192 [==============================] - 139s 722ms/step - loss: 1.8396e-04 - accuracy: 1.0000 - val_loss: 0.0307 - val_accuracy: 0.9920
    Epoch 12/25
    192/192 [==============================] - 127s 660ms/step - loss: 1.5278e-04 - accuracy: 1.0000 - val_loss: 0.0306 - val_accuracy: 0.9923
    Epoch 13/25
    192/192 [==============================] - 126s 657ms/step - loss: 1.2209e-04 - accuracy: 1.0000 - val_loss: 0.0308 - val_accuracy: 0.9923
    Epoch 14/25
    192/192 [==============================] - 130s 676ms/step - loss: 1.0745e-04 - accuracy: 1.0000 - val_loss: 0.0307 - val_accuracy: 0.9925
    Epoch 15/25
    192/192 [==============================] - 131s 683ms/step - loss: 9.1814e-05 - accuracy: 1.0000 - val_loss: 0.0310 - val_accuracy: 0.9925
    Epoch 16/25
    192/192 [==============================] - 132s 686ms/step - loss: 8.1444e-05 - accuracy: 1.0000 - val_loss: 0.0312 - val_accuracy: 0.9923
    Epoch 17/25
    192/192 [==============================] - 154s 805ms/step - loss: 7.1451e-05 - accuracy: 1.0000 - val_loss: 0.0310 - val_accuracy: 0.9928
    Epoch 18/25
    192/192 [==============================] - 175s 913ms/step - loss: 6.0205e-05 - accuracy: 1.0000 - val_loss: 0.0314 - val_accuracy: 0.9924
    Epoch 19/25
    192/192 [==============================] - 199s 1s/step - loss: 5.3026e-05 - accuracy: 1.0000 - val_loss: 0.0317 - val_accuracy: 0.9922
    Epoch 20/25
    192/192 [==============================] - 173s 901ms/step - loss: 4.7099e-05 - accuracy: 1.0000 - val_loss: 0.0319 - val_accuracy: 0.9923
    Epoch 21/25
    192/192 [==============================] - 204s 1s/step - loss: 3.8788e-05 - accuracy: 1.0000 - val_loss: 0.0321 - val_accuracy: 0.9923
    Epoch 22/25
    192/192 [==============================] - 174s 903ms/step - loss: 3.6134e-05 - accuracy: 1.0000 - val_loss: 0.0322 - val_accuracy: 0.9925
    Epoch 23/25
    192/192 [==============================] - 171s 889ms/step - loss: 3.0264e-05 - accuracy: 1.0000 - val_loss: 0.0324 - val_accuracy: 0.9924
    Epoch 24/25
    192/192 [==============================] - 159s 828ms/step - loss: 2.7475e-05 - accuracy: 1.0000 - val_loss: 0.0328 - val_accuracy: 0.9923
    Epoch 25/25
    192/192 [==============================] - 150s 783ms/step - loss: 2.4587e-05 - accuracy: 1.0000 - val_loss: 0.0329 - val_accuracy: 0.9923
    


```python
# Generating generalization metric  s
score = model.evaluate(input_test, target_test, verbose=0)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
```

    Test loss: 0.03310513123869896 / Test accuracy: 0.9918000102043152
    


```python

```
