from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
import numpy as np

(x_train, y_train),(x_test, y_test) = fashion_mnist.load_data()

x_train=np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test=np.reshape(x_test, (len(x_test), 28, 28, 1))
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

ip = Input(shape=(28, 28, 1))
x = Conv2D(16, (3, 3), activation='relu', padding='same')(ip)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dense(10, activation='softmax')(x)
x = Model(ip, x)


x.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy'])
x.fit(x_train, y_train, epochs = 10, batch_size = 128, shuffle = True, verbose = 1, validation_data = (x_test, y_test))
x.save("fashion_mnist2.h5")