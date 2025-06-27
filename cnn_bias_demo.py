import streamlit as st
st.set_page_config(layout="wide", page_title="Digit Classifier")

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
#  CPU setup only
# ─────────────────────────────────────────────────────────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU only
st.info("Forcing CPU-only training. GPU is disabled.")

# ─────────────────────────────────────────────────────────────────────────────
#  Helpers to save / load models
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model_from_path(model_path: str):
    return keras.models.load_model(model_path)

# ─────────────────────────────────────────────────────────────────────────────
#  Load MNIST
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_mnist_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x = np.concatenate([x_train, x_test], axis=0).astype("float32") / 255.0
    y = np.concatenate([y_train, y_test], axis=0).astype("int32")
    x = np.expand_dims(x, axis=-1)
    return x, y

# ─────────────────────────────────────────────────────────────────────────────
#  Data Augmentation
# ─────────────────────────────────────────────────────────────────────────────
def get_augmentation_layer():
    return keras.Sequential([
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomFlip("horizontal"),
    ], name="augmentation")

# ─────────────────────────────────────────────────────────────────────────────
#  LeNet-5 for Digits
# ─────────────────────────────────────────────────────────────────────────────
def build_lenet5(num_classes: int):
    inputs = keras.Input(shape=(28, 28, 1))
    x = get_augmentation_layer()(inputs)
    x = layers.Conv2D(6, 5, activation="tanh", padding="same")(x)
    x = layers.AveragePooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(16, 5, activation="tanh")(x)
    x = layers.AveragePooling2D(pool_size=(2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(120, activation="tanh")(x)
    x = layers.Dense(84, activation="tanh")(x)
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)
    model = keras.Model(inputs, outputs, name="lenet5")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model(tf.zeros((1, 28, 28, 1)))  # Ensure model is built immediately
    return model

# ─────────────────────────────────────────────────────────────────────────────
#  Preprocessing
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_digit_image(uploaded_image):
    pil = Image.open(uploaded_image).convert("L").resize((28, 28))
    arr = np.array(pil).astype("float32")
    if arr.mean() > 127:
        arr = 255.0 - arr
    arr = arr / 255.0
    arr = np.expand_dims(arr, axis=-1)
    return np.expand_dims(arr, axis=0), pil

# ─────────────────────────────────────────────────────────────────────────────
#  Prediction bar plot
# ─────────────────────────────────────────────────────────────────────────────
def plot_prediction_probs(preds):
    fig, ax = plt.subplots()
    bars = ax.bar(range(len(preds[0])), preds[0], color="skyblue")
    pred_idx = int(np.argmax(preds[0]))
    conf = preds[0][pred_idx] * 100
    bars[pred_idx].set_color("dodgerblue")
    ax.set_xticks(range(len(preds[0])))
    ax.set_xticklabels([str(i) for i in range(len(preds[0]))])
    ax.set_xlabel("Digit")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    ax.set_title(f"🎯 Prediction: {pred_idx} ({conf:.2f}%)", fontsize=16)
    st.pyplot(fig)

# ─────────────────────────────────────────────────────────────────────────────
#  Streamlit UI
# ─────────────────────────────────────────────────────────────────────────────
st.title("LeNet-5 Digit Classifier (MNIST 0–9)")
epochs = st.sidebar.slider("Training Epochs", 1, 20, 5)
num_classes = st.sidebar.slider("Number of Digits to Train (0 to N-1)", 1, 10, 10)
model_path = f"models/lenet5_digits_{num_classes}.h5"

st.header("▶️ Train / Evaluate Digit-Only Model (MNIST 0–9)")
if st.button("Train Digit Model"):
    with st.spinner("Loading MNIST data..."):
        x_d, y_d = load_mnist_data()
        mask = y_d < num_classes
        x_d, y_d = x_d[mask], y_d[mask]

    idx = np.arange(len(x_d))
    np.random.shuffle(idx)
    x_d, y_d = x_d[idx], y_d[idx]
    split = int(0.9 * len(x_d))
    x_train, y_train = x_d[:split], y_d[:split]
    x_val, y_val = x_d[split:], y_d[split:]
    with st.spinner(f"Training for {epochs} epochs on {num_classes} classes..."):
        model = build_lenet5(num_classes)
        model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs, batch_size=128, verbose=1)
    os.makedirs("models", exist_ok=True)
    model.save(model_path)
    st.success(f"✅ Digit model trained and saved to `{model_path}`.")

uploaded = st.file_uploader("Upload a digit image (0–9)", type=["png", "jpg", "jpeg"])
if uploaded:
    arr, pil_img = preprocess_digit_image(uploaded)
    st.image(pil_img, caption="Uploaded Image", use_column_width=True)
    if not os.path.exists(model_path):
        st.error("Model not found. Train first.")
        st.stop()
    model = load_model_from_path(model_path)
    _ = model(arr, training=False)
    preds = model.predict(arr)
    st.markdown(f"📦 Loaded Model: `{model_path}`")
    st.markdown(f"✅ Model trained to classify digits from `0` to `{num_classes - 1}`.")
    plot_prediction_probs(preds)
