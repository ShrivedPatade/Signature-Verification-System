import streamlit as st
import numpy as np

from PIL import Image

import cv2

import tensorflow as tf
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate, Dropout, BatchNormalization, MaxPooling2D, Concatenate, Lambda, Flatten, Dense
from keras.models import Model

from keras.initializers import glorot_uniform

from tensorflow.keras.layers import Layer
from keras.regularizers import l2
import tensorflow.keras.backend as K

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

def create_base_network_signet(input_shape):
    seq = Sequential()
    seq.add(Conv2D(96, kernel_size=(11, 11), activation='relu', name='conv1_1', strides=4, input_shape= input_shape,
                        kernel_initializer='glorot_uniform', data_format='channels_last')) # Assuming 'tf' dim_ordering means channels_last
    seq.add(BatchNormalization(epsilon=1e-06, axis=1, momentum=0.9)) # mode argument is deprecated
    seq.add(MaxPooling2D((3,3), strides=(2, 2)))
    seq.add(ZeroPadding2D((2, 2), data_format='channels_last')) # Changed dim_ordering to data_format

    seq.add(Conv2D(256, kernel_size=(5, 5), activation='relu', name='conv2_1', strides=1, kernel_initializer='glorot_uniform',  data_format='channels_last')) # Changed init and dim_ordering
    seq.add(BatchNormalization(epsilon=1e-06, axis=1, momentum=0.9)) # mode argument is deprecated
    seq.add(MaxPooling2D((3,3), strides=(2, 2)))
    seq.add(Dropout(0.3))# added extra
    seq.add(ZeroPadding2D((1, 1), data_format='channels_last')) # Changed dim_ordering to data_format

    seq.add(Conv2D(384, kernel_size=(3, 3), activation='relu', name='conv3_1', strides=1, kernel_initializer='glorot_uniform',  data_format='channels_last')) # Changed init and dim_ordering
    seq.add(ZeroPadding2D((1, 1), data_format='channels_last')) # Changed dim_ordering to data_format

    seq.add(Conv2D(256, kernel_size=(3, 3), activation='relu', name='conv3_2', strides=1, kernel_initializer='glorot_uniform', data_format='channels_last'))  # Changed init and dim_ordering
    seq.add(MaxPooling2D((3,3), strides=(2, 2)))
    seq.add(Dropout(0.3))# added extra
    seq.add(Flatten(name='flatten'))
    seq.add(Dense(1024, kernel_regularizer=l2(0.0005), activation='relu', kernel_initializer='glorot_uniform')) # Changed W_regularizer to kernel_regularizer and init to kernel_initializer
    seq.add(Dropout(0.5))

    seq.add(Dense(128, kernel_regularizer=l2(0.0005), activation='relu', kernel_initializer='glorot_uniform')) # softmax changed to relu, Changed W_regularizer to kernel_regularizer and init to kernel_initializer

    return seq

input_shape=(280, 280, 1)

base_network = create_base_network_signet(input_shape)

input_a = Input(shape=(input_shape))
input_b = Input(shape=(input_shape))
processed_a = base_network(input_a)
processed_b = base_network(input_b)

distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])

model = Model(inputs=[input_a, input_b], outputs=distance)

rms = RMSprop(learning_rate=1e-4, rho=0.9, epsilon=1e-08)
model.compile(loss=contrastive_loss, optimizer=rms)
model.load_weights('net-020.weights.h5')

def preprocess(img, t):
  _, thresh = cv2.threshold(img, t, 255, cv2.THRESH_BINARY)
  rs = cv2.resize(thresh, (280, 280))
  inv = cv2.bitwise_not(rs)
  inv = np.array(inv, dtype = np.float64)
  inv /= 255
  return inv

st.set_page_config(page_title='Proxy Detect | CNN')
st.title("Signature Verification System")

# Create two columns with more space between them
col1, _, col3 = st.columns([0.4, 0.1, 0.4])

with col1:
    t = st.number_input("Threshold Value", min_value=0, max_value=255, value=220)
    uploaded_files = st.file_uploader("Upload Signature Image...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files is not None:
        if len(uploaded_files) == 2:
            images = []
            for uploaded_file in uploaded_files:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
                image = preprocess(image, t)
                images.append(image)

            st.image([np.squeeze(i) for i in images], caption=["Known Real", "Not Known"], use_column_width=True)
        else:
            st.warning("Please upload exactly two images.")

with col3:
    st.header("Results - CNN")
    if st.button("Run"):
        if t >= 220:
            thres = 1.49
        else:
            thres = 0.15
        img1 = np.expand_dims(images[0], axis=0)
        img2 = np.expand_dims(images[1], axis=0)
        res = model.predict([img1, img2], verbose=0)
        diff = res[0][0]
        if diff > 1.49:
            st.header("Signature is a proxy")
        else:
            st.header("Signature is real")
        st.write(f"The difference score is: {diff}")