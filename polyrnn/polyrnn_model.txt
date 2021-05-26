import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, ConvLSTM2D, TimeDistributed
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model

### modified VGG16

_input = Input((224,224,1))

conv1  = Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu")(_input)
conv2  = Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu")(conv1)
pool1  = MaxPooling2D((2, 2))(conv2)

conv3  = Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")(pool1)
conv4  = Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")(conv3)
pool2  = MaxPooling2D((2, 2))(conv4)
pool_1 = MaxPooling2D((2, 2))(pool2)
conv_1  = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(pool_1)

conv5  = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(pool2)
conv6  = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(conv5)
conv7  = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(conv6)
pool3  = MaxPooling2D((2, 2))(conv7)
conv_2  = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(pool3)

conv8  = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(pool3)
conv9  = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(conv8)
conv10 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(conv9)
pool4  = MaxPooling2D((2, 2))(conv10)
conv_3 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(conv10)

conv11 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(pool4)
conv12 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(conv11)
conv13 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(conv12)
pool5  = MaxPooling2D((2, 2))(conv13)
conv_4 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(conv13)
up_1 = UpSampling2D(size=(2, 2))(conv_4)

fused = tf.keras.layers.add([conv_1, conv_2, conv_3, up_1])

output = Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")(fused)

#RNN decoder 

convltsm_1 = ConvLSTM2D(filters=64, kernel_size=(3,3), activation='relu', return_sequences = True)(output)
convltsm_2 = ConvLSTM2D(filters=16, kernel_size=(3,3), activation='relu', return_sequences = False)(convltsm_1)
dropout = Dropout(0.5)(convltsm_2)
flatten = Flatten()(dropout)
dense_1 = Dense(100, activation='relu')(flatten)
dense_2 = TimeDistributed(Dense(10, activation='softmax'))

polyrnn_model  = Model(inputs=_input, outputs=dense_2)
