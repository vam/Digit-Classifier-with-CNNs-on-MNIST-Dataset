import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import pandas as pd

# Function to preprocess uploaded image
def preprocess_image(uploaded_file):
    img = image.load_img(uploaded_file, target_size=(28, 28), color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Function to retrieve incorrect predictions
import re

def retrieve_incorrect_predictions(folder='incorrect_predictions'):
    incorrect_predictions = []

    if os.path.exists(folder):
        for filename in os.listdir(folder):
            if filename.endswith('.jpg'):
                filepath = os.path.join(folder, filename)
                # Extract information from the filename
                match = re.match(r'incorrect_prediction_(\d+)_predicted(\d+)_actual(\d+).jpg', filename)
                
                if match:
                    timestamp, predicted_class, actual_label = map(int, match.groups())
                    
                    incorrect_predictions.append({
                        'filepath': filepath,
                        'timestamp': timestamp,
                        'predicted_class': predicted_class,
                        'actual_label': actual_label
                    })
                else:
                    print(f"Invalid filename format: {filename}")

    return incorrect_predictions



# Function to fine-tune the model using incorrect predictions
def fine_tune_model(model, incorrect_predictions):
    for example in incorrect_predictions:
        img = preprocess_image(example['filepath'])
        actual_label = example['actual_label']

        # Create one-hot encoding for the actual label
        actual_label_one_hot = np.zeros((1, num_classes))
        actual_label_one_hot[0, actual_label] = 1

        # Fine-tune the model using the incorrect prediction example
        model.train_on_batch(img, actual_label_one_hot)

    return model

# Function to evaluate the model on test data
def evaluate_model(model, test_data, test_labels):
    # Evaluate the model
    evaluation = model.evaluate(test_data, test_labels)
    print("Test Accuracy:", evaluation[1])

# Function to load test data from CSV and preprocess images
def load_test_data(csv_path, test_fraction=0.2):
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_path)

    # Split the data into training and test sets
    train_df, test_df = train_test_split(df, test_size=test_fraction, random_state=42)

    # Extract images and labels from the test set
    test_images = np.array([preprocess_image_from_csv(row) for _, row in test_df.iterrows()])
    test_labels = to_categorical(test_df['label'], num_classes=10)  # Assuming you have 10 classes for digits

    return test_images, test_labels

# Function to preprocess image from CSV row
def preprocess_image_from_csv(row):
    # Extract pixel values from the row and reshape into (28, 28, 1) image
    img_array = row.values[1:].astype(np.uint8).reshape(28, 28, 1)
    img_array = img_array / 255.0  # Normalize pixel values
    return img_array

# Assuming you have a dataset and labels for testing
csv_path = 'train.csv'
test_data, test_labels = load_test_data(csv_path)

# Set the number of classes in your problem
num_classes = 10

# Load your pretrained model
pretrained_model_path = 'digit_classifier_model.h5'
pretrained_model = load_model(pretrained_model_path)

# Retrieve incorrect predictions
incorrect_predictions = retrieve_incorrect_predictions()

print(incorrect_predictions)
model1 = fine_tune_model(pretrained_model, incorrect_predictions)

# Save the fine-tuned model
finetuned_model_path = 'digit_classifier_model.h5'
model1.save(finetuned_model_path)

# Evaluate the model
evaluate_model(model1, test_data, test_labels)


  

