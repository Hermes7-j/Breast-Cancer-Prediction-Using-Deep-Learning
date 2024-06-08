import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the saved model
model = tf.keras.models.load_model("cancer_model.h5")

# Streamlit app
st.title("Breast Cancer Detection App")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    # Preprocess the image
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Make predictions
    prediction = model.predict(img_array)

    # Display the results
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Generalized classification
    if prediction[0][0] > 0.5:
        st.write("Result: Disease")
        st.write("Specific Classification: Malignant" if prediction[0][0] > 0.5 else "Benign")
    else:
        st.write("Result: No Disease")

    
