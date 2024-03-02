import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
from datetime import datetime
import numpy as np

# Load your trained model
model_path = 'digit_classifier_model.h5'
model = load_model(model_path)

# Flag to track whether the prediction has already been saved
prediction_saved = False


# =================================== Title ===============================
st.title("""
MNIST Image classification model for AI-ML Lab Assignment 1
B Vamshi Krishna  (12140410)
    """)

# ================================ About =================================
st.write("""
## 1️⃣ About
    """)
st.write("""
Hi all, Welcome to this project. It is a MNIST Classification App developed for my AI/ML Lab Course!!!
    """)

# ============================ How To Use It ===============================
st.write("""
## 2️⃣ How To Use It
    """)
st.write("""
Well, it's pretty simple!!!
- Let me clear first, the model has the power to predict the image of Handwritten Digits, so you are requested to give an image of a digit.
- First of all, download the image of a digit.
- Next, just Browse that file or Drag & drop that file.
- Please make sure that you are uploading a picture file.
- it can predict which digit's image you have uploaded.

🔘 **NOTE :** *If you upload other than an image file, then it will show an error message when you select the File!!!*
    """)

# ========================= What It Will Predict ===========================
st.write("""
## 3️⃣ What It Will Predict
    """)
st.write("""
Well, it can predict which digit's image you have uploaded, even if it is handwritten!
    """)


# Streamlit app
def main():
    global prediction_saved
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image_display = Image.open(uploaded_file)
        st.image(image_display, caption="Uploaded Image.", use_column_width=True)
        st.write("")

        # Preprocess the image
        img = preprocess_image(uploaded_file)

        # Display prediction
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction)
        st.write(f"Prediction: {predicted_class}")

        # Feedback interface
        feedback = st.radio("Was the prediction correct?", ('Correct', 'Incorrect'))

        # Store incorrect predictions
        if feedback == 'Incorrect':
            actual_label = st.text_input("Enter the actual label:")
            if actual_label is not None:
                fb = st.button('submit feedback')
                if fb:
                    save_incorrect_prediction(uploaded_file, predicted_class, actual_label,feedback)
            

def preprocess_image(uploaded_file):
    img = image.load_img(uploaded_file, target_size=(28, 28), color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def save_incorrect_prediction(uploaded_file, predicted_class, actual_label, feedback, output_directory='incorrect_predictions'):
    # Check if the feedback is 'Incorrect'
    if feedback == 'Incorrect':
        # Create output directory if it doesn't exist
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        # Generate a unique filename based on timestamp, predicted class, and actual label
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"incorrect_prediction_{timestamp}_predicted{predicted_class}_actual{actual_label}.jpg"
        filepath = os.path.join(output_directory, filename)

        # Save the image file
        with open(filepath, 'wb') as f:
            # Reset the file pointer to the beginning of the file
            uploaded_file.seek(0)
            f.write(uploaded_file.read())



if __name__ == "__main__":
    main()

