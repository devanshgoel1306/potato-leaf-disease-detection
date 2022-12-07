import streamlit as st
#for defining the loss in compile function
import tensorflow as tf

#for loading the model trained earlier for prediction purpose
from tensorflow.keras.models import load_model

#for loading the image and making changes in it
#to make it fit for prediction purpose
from tensorflow.keras.preprocessing import image

import numpy as np

#for adding background image
import base64
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('back.jpeg')

#adding title
st.title('POTATO LEAF DISEASE DETECTOR')

# target dimensions of image
img_width, img_height = 256, 256

#for uploading the image
test= st.file_uploader("",type= ["png","jpg","jpeg"])

if test!=None and st.button('Predict'):
    # load the model we saved
    model = load_model('potatoes.h5')
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  optimizer='adam',
                  metrics=['accuracy'])

    # predicting images
    #creating an object of image and changing its size to required size
    img = image.load_img(test, target_size=(img_width, img_height))

    #converting the image to array
    x = image.img_to_array(img)


    x = np.expand_dims(x, axis=0)
    image = np.vstack([x])
    
    #gives probability for each class
    prob = model.predict(image)

    #possible classes
    class_= ['Potato Early Blight', 'Potato Late Blight', 'Potato Healthy']

    #print class with maximum probability
    st.metric(label= "Predicted Class", value= class_[np.argmax(prob)])
