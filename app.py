import streamlit as st
import torch
from PIL import Image
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image as keras_image

# Function to load YOLOv5 model
@st.cache(allow_output_mutation=True)
def load_model_yolov5(weights_path):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)
    return model

# Function to perform inference on uploaded image using YOLOv5
def predict_yolov5(image, model):
    results = model(image)  # Inference
    return results

# Function to create emotion prediction model
def create_model_emotion():
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3), pooling='avg')
    base_model.trainable = False
    model = Sequential([
        base_model,
        BatchNormalization(),
        Dense(256, activation='relu'),
        Dropout(0.4),
        Dense(4, activation='softmax')  
    ])
    model.build((None, 224, 224, 3)) 
    return model

# Load emotion prediction model
model_emotion = create_model_emotion()
model_emotion.load_weights('animal_model.weights.h5')

# Load YOLOv5 model
weights_path_yolov5 = 'best.pt'  
model_yolov5 = load_model_yolov5(weights_path_yolov5)

# Load class labels for emotion prediction
class_labels_emotion = ['Angry', 'Other', 'Sad', 'Happy']

# Function to predict emotion
def predict_emotion(img_path):
    img = keras_image.load_img(img_path, target_size=(224, 224))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    prediction = model_emotion.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    predicted_class_label = class_labels_emotion[predicted_class_index]

    return predicted_class_label

def main():
    st.title('Animal Detection and Emotion Prediction')

    # File upload section
    st.sidebar.title('Upload Image')
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        results_yolov5 = predict_yolov5(image, model_yolov5)

        st.subheader('Animal Detection Results:')
        for detection in results_yolov5.xyxy[0]:
            class_id = int(detection[5])
            class_name = model_yolov5.names[class_id]
            confidence = float(detection[4])
            st.write(f"Class: {class_name}, Confidence: {confidence:.2f}")

        temp_file = 'temp.jpg'
        with open(temp_file, "wb") as f:
            f.write(uploaded_file.getbuffer())

        emotion = predict_emotion(temp_file)

        st.subheader('Emotion Prediction:')
        st.write(f"The predicted emotion of the animal is: **{emotion}**")

if __name__ == '__main__':
    main()
