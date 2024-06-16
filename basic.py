import streamlit as st
import torch
from PIL import Image
import os

# Function to load YOLOv5 model
@st.cache(allow_output_mutation=True)
def load_model(weights_path):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)
    return model

# Function to perform inference on uploaded image
def predict(image, model):
    results = model(image)  # Inference
    return results

def main():
    st.title('Animal Detection with YOLOv5')

    # File upload and model loading section
    st.sidebar.title('Upload Image')
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        # Load YOLOv5 model
        weights_path = 'best.pt'  # Update with your path to best.pt
        model = load_model(weights_path)

        # Perform inference
        results = predict(image, model)

        # Display results
        st.subheader('Detection Results:')
        for detection in results.xyxy[0]:
            class_id = int(detection[5])
            class_name = model.names[class_id]
            confidence = float(detection[4])
            st.write(f"Class: {class_name}, Confidence: {confidence:.2f}")

if __name__ == '__main__':
    main()
