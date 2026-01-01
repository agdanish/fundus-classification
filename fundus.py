import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import cv2
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import time
from keras.applications import InceptionV3

custom_css = """
    body {
        background-color: #f0f0f0;
    }
    .st-emotion-cache-183lzff.exotz4b0 {
        color: blue;
    }
"""

icon = Image.open("img/1.jfif")
st.set_page_config(
    page_title="fundus",
    page_icon=icon,
)
class_names = ['AbNormal', 'Normal']

# Load model without compilation to avoid version compatibility issues
model = tf.keras.models.load_model("fun.h5", compile=False)

# Recompile the model with current Keras version
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

target_size = (150, 150)

pretrained_model= InceptionV3(include_top=False,
                   input_shape=(150,150,3),
                   pooling='avg',classes=2,
                   weights='imagenet')
print(len(pretrained_model.layers))
pretrained_model.summary()

# Sidebar
with st.sidebar:
    st.image(icon)
    st.subheader("FUNDUS")
    st.caption("PREDICTION")
    st.markdown(f'<style>{custom_css}</style>', unsafe_allow_html=True) 
    st.text("Number of layers in InceptionV3 \nmodel:"+str(len(pretrained_model.layers)))
    st.text("Along InceptionV3, it consists \nof "+str(len(model.layers))+" layers")
    st.subheader(":arrow_up: UPLOAD IMAGE")
    uploaded_file = st.file_uploader("FUNDUS")


# Body
image_path = "img/5.jpg"
st.header("FUNDUS")
st.image(image_path, width=800)  

col1, col2 = st.columns(2)
y_pred = None


def predict_image(img):
    # Resize the input image to 150x150
    img_resized = cv2.resize(img, (150, 150))
    img_3d = img_resized.reshape(1, 150, 150, 3)
    start = time.time()
    prediction = model.predict(img_3d)[0]
    end_predict = time.time()
    res= {class_names[i]: round(prediction[i]*100) for i in range(2)}
    Keymax = max(zip(res.values(), res.keys()))[1]
    st.subheader(":white_check_mark: PREDICTION")
    st.subheader(Keymax)   
    res2= {class_names[i]:prediction[i]*100 for i in range(2)}
    my_list_key = list(res2.keys())
    my_list_val = list(res2.values())
    with col2:

        st.subheader("Time to predict: "+"{:.2f}".format(end_predict-start)+" s")
        st.subheader(":bar_chart: Analysis")
        fig, ax = plt.subplots(figsize=(5, 2))
        ax.barh(class_names, my_list_val, height=0.55, align="center")
        for i, (c, p) in enumerate(zip(class_names, my_list_val)):
            ax.text(p + 2, i - 0.2, f"{p:.2f}%")
        ax.grid(axis="x")
        ax.set_xlim([0, 120])
        ax.set_xticks(range(0, 101, 20))
        fig.tight_layout()
        st.pyplot(fig)

if uploaded_file is not None:
    with col1:
        st.subheader(":camera: INPUT")
        st.image(uploaded_file, use_column_width=True)

        img = tf.keras.utils.load_img(
            uploaded_file, target_size=target_size
        )
        
        img = tf.keras.utils.img_to_array(img)
        img_aux = img.copy()

    
        if st.button(
            ":arrows_counterclockwise: PREDICT"
        ):
            with col2:
                img_array = np.expand_dims(img_aux, axis=0)
                img_array = np.float32(img_array)
                img_array = tf.keras.applications.xception.preprocess_input(
                    img_array
                )
            
                predict_image(img)
