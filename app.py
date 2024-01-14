import streamlit as st
import os
from PIL import Image
from keras.preprocessing import image
from keras.models import load_model
from keras import models
from keras import layers
from datetime import datetime
from keras.preprocessing import image as keras_image
import numpy as np






def get_prediction(image_path):
    your_target_size = (150,150,3)
    img = keras_image.load_img(image_path,target_size=(your_target_size))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    model = load_model('vgg16.h5')
    prediction = model.predict(img_array)
    return prediction

def save_image(image, save_folder="Images_Directory"):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    image.save(os.path.join(save_folder, f"uploaded_image_{timestamp}.jpg"))


def display_output(predictions):
    class_label = "Formal" if predictions[0][0] < 0.5 else "Informal"
    confidence = predictions[0][0]

    # Display the prediction result with creativity
    st.subheader("Prediction Results:")



    if class_label == "Formal":
        st.markdown(
            f'<div style="background-color: #4CAF50; color: white; padding: 10px; text-align: center; border-radius: 10px;">'
            f'<h4>This is Formal Wear.</h4>'
            f'<p>🌟 Dressed to impress! This outfit exudes sophistication and is perfect for a formal affair.</p>'
            f'</div>', unsafe_allow_html=True
        )
        st.balloons()


    else:
        st.markdown(
            f'<div style="background-color: #4B0082; color: white; padding: 10px; text-align: center; border-radius: 10px;">'
            f'<h4>This is InFormal Wear.</h4>'
            f'<p>🌈 Casual and carefree! This outfit is ideal for a relaxed day or a fun get-together with friends.</p>'
            f'</div>', unsafe_allow_html=True
        )
        st.balloons()


    st.write("")

    # Add additional creative elements
    st.success("👗 Fashion is a form of self-expression. Wear it with confidence and let your style speak volumes")
    st.info("Try another image to see more styles.")


st.title("WearItRight: Dress Code Classifier")
st.write("👔 Whether formal or casual, every outfit tells a unique story. What story does your style tell?")
uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_column_width=False,width=500)

    st.header("Make Prediction")
    if st.button("Make Prediction"):
        save_image(image)
        latest_image_path = "Images_Directory/" + sorted(os.listdir("Images_Directory"))[-1]
        predictions = get_prediction(latest_image_path)
        display_output(predictions)