import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import streamlit as st
from PIL import Image
import tensorflow_datasets as tfds

# Ensure GPU is utilized (if available)
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    tf.keras.mixed_precision.set_global_policy('mixed_float16')  # Enable mixed precision for speedup
else:
    st.warning("GPU not detected. Using CPU for training.")

# Cache the model loading
@st.cache_resource
def load_model(model_path):
    return keras.models.load_model(model_path)

# Define class names for each dataset
class_names_map = {
    "Numbers": [str(i) for i in range(7)],  # "0" to "6"
    "Shapes": ["Circle", "Square", "Triangle", "Star", "Heart", "Diamond", "Pentagon", "Hexagon", "Octagon", "Cross"],
    "Alphabets": [chr(i) for i in range(65, 91)]  # "A" to "Z"
}

# Efficient dataset loading using TensorFlow datasets
@st.cache_data
def load_numbers():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Ensure grayscale shape (28,28,1)
    x_train = np.expand_dims(x_train, axis=-1).astype('float32')
    x_test = np.expand_dims(x_test, axis=-1).astype('float32')

    # Filter for digits 0-6
    mask_train, mask_test = y_train < 7, y_test < 7
    return x_train[mask_train], y_train[mask_train], x_test[mask_test], y_test[mask_test]

@st.cache_data
def load_shapes():
    ds = tfds.load('quickdraw_bitmap', split='train', as_supervised=False)
    ds = ds.shuffle(100000).take(50000)
    images, labels = [], []

    for example in ds:
        img = example['image']
        label = example['label']

        img_resized = tf.image.resize(img, (28, 28)).numpy().astype('float32') / 255.0
        images.append(img_resized)
        labels.append(label.numpy())

    labels = np.clip(np.array(labels), 0, 9)  # Ensure labels are within 0-9 range
    return np.expand_dims(np.array(images), axis=-1), labels  # Ensure (28,28,1)

@st.cache_data
def load_alphabets():
    ds = tfds.load('emnist/letters', split='train', as_supervised=False)
    ds = ds.shuffle(100000).take(50000)
    images, labels = [], []

    for example in ds:
        img = example['image']
        label = example['label']

        # Rotate 90° and flip horizontally to correct orientation
        img = tf.image.rot90(img)
        img = tf.image.flip_left_right(img)

        img_resized = tf.image.resize(img, (28, 28)).numpy().astype('float32') / 255.0
        images.append(img_resized)
        labels.append(label.numpy() - 1)  # Adjust label range to 0-25

    return np.expand_dims(np.array(images), axis=-1), np.array(labels)  # Ensure (28,28,1)

# Data augmentation for robustness
data_augmentation = keras.Sequential([
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomFlip("horizontal"),
])

# Updated CNN model with augmentation layer
def create_model(num_classes):
    model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),  # Ensure correct input shape
        data_augmentation,  # Apply augmentation
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax', dtype='float32')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to preprocess uploaded images
def preprocess_uploaded_image(uploaded_image):
    image = Image.open(uploaded_image).convert('L')  # Convert to grayscale
    image = image.resize((28, 28))  # Resize
    image_array = np.array(image) / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=-1).astype('float32')  # Add channel dimension
    return np.expand_dims(image_array, axis=0)  # Add batch dimension

# Main Streamlit interface
st.title("CNN Bias Demonstration")
dataset_option = st.selectbox("Select Dataset for Training", ["Numbers", "Shapes", "Alphabets"])

# Define number of classes per dataset
num_classes_map = {"Numbers": 7, "Shapes": 10, "Alphabets": 26}

if st.button("Train Model"):
    with tf.device('/GPU:0' if physical_devices else '/CPU:0'):
        if dataset_option == "Numbers":
            x_train, y_train, x_test, y_test = load_numbers()
        elif dataset_option == "Shapes":
            x_train, y_train = load_shapes()
        else:
            x_train, y_train = load_alphabets()

        model_path = f"models/cnn_{dataset_option.lower()}.h5"
        model = create_model(num_classes_map[dataset_option])

        if dataset_option == "Numbers":
            model.fit(x_train, y_train, epochs=12, validation_data=(x_test, y_test))
        else:
            model.fit(x_train, y_train, epochs=12, batch_size=32, validation_split=0.1)

        model.save(model_path)
        st.success("Training Completed!")

# Upload and compare images
st.subheader("Upload Your Own Image for Comparison")
uploaded_image = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_image:
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    st.write("Classifying the uploaded image...")

    # Preprocess and predict
    img_array = preprocess_uploaded_image(uploaded_image)
    model_path = f"models/cnn_{dataset_option.lower()}.h5"

    # If model is already saved, load it, else train a new one
    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        st.error("Model not found. Please train the model first.")
        st.stop()

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_prob = np.max(predictions) * 100

    # Get class name
    class_names = class_names_map[dataset_option]

    # Adjust for Alphabets (Add 1 back to predicted class)
    if dataset_option == "Alphabets":
        predicted_class += 1

    predicted_class_name = class_names[predicted_class]  # Adjust back to 0-indexed class names

    st.write(f"Predicted Class: **{predicted_class_name}**")
    st.write(f"Confidence: **{predicted_prob:.2f}%**")

    # Display top 3 predictions
    st.write("Class probabilities:")
    sorted_indices = np.argsort(predictions[0])[::-1][:3]
    for i in sorted_indices:
        st.write(f"**{class_names[i]}**: {predictions[0][i] * 100:.2f}%")
