import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import streamlit as st
from PIL import Image
import tensorflow_datasets as tfds

# ─────────────────────────────────────────────────────────────────────────────
#  GPU setup
# ─────────────────────────────────────────────────────────────────────────────
physical_devices = tf.config.list_physical_devices("GPU")
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
else:
    st.warning("GPU not detected. Using CPU for training.")

# ─────────────────────────────────────────────────────────────────────────────
#  Helpers to save / load models
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model_from_path(model_path: str):
    return keras.models.load_model(model_path)

# ─────────────────────────────────────────────────────────────────────────────
#  1) “Digits” Model: load MNIST, train a LeNet-5 on 10 classes (0–9)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_mnist_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x = np.concatenate([x_train, x_test], axis=0).astype("float32") / 255.0
    y = np.concatenate([y_train, y_test], axis=0).astype("int32")
    x = np.expand_dims(x, axis=-1)  # (70000,28,28,1)
    return x, y

# ─────────────────────────────────────────────────────────────────────────────
#  2) “Letters” Model: load EMNIST/letters, rotate+flip exactly as in train
#     Map A→0, B→1, …, Z→25. (26 classes total)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_emnist_letters_data():
    # Each example['label'] ∈ {1..26}. We want {0..25}.
    ds = tfds.load("emnist/letters", split="train", as_supervised=False)
    ds = ds.shuffle(100_000).take(60_000)  # 60k examples for training

    images = []
    labels = []
    for example in tfds.as_numpy(ds):
        img = example["image"]  # (28,28,1), uint8
        # Rotate CW 90° + flip left↔right to match MNIST orientation
        pil = Image.fromarray(img.reshape(28, 28).astype("uint8"))
        pil = pil.rotate(-90, expand=False)
        pil = pil.transpose(Image.FLIP_LEFT_RIGHT)
        arr = np.array(pil).astype("float32") / 255.0  # (28,28)
        arr = np.expand_dims(arr, axis=-1)            # (28,28,1)
        images.append(arr)
        labels.append(int(example["label"]) - 1)       # shift 1..26 → 0..25

    x = np.stack(images, axis=0)                       # (60000,28,28,1)
    y = np.array(labels, dtype="int32")                # (60000,)
    return x, y

# ─────────────────────────────────────────────────────────────────────────────
#  Data‐augmentation block (shared, but you can tweak separately if desired)
# ─────────────────────────────────────────────────────────────────────────────
data_augmentation = keras.Sequential(
    [
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomFlip("horizontal"),
    ]
)

# ─────────────────────────────────────────────────────────────────────────────
#  LeNet-5 for digits (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
def build_lenet5(num_classes: int):
    model = keras.Sequential(
        [
            layers.Input(shape=(28, 28, 1)),
            # (optional) small augmentation for digits (can be commented out)
            data_augmentation,
            # LeNet-5 conv/pool block 1
            layers.Conv2D(6, kernel_size=5, strides=1, activation="tanh", padding="same"),
            layers.AveragePooling2D(pool_size=(2, 2)),  # → 14×14×6
            # LeNet-5 conv/pool block 2
            layers.Conv2D(16, kernel_size=5, strides=1, activation="tanh"),
            layers.AveragePooling2D(pool_size=(2, 2)),  # → 5×5×16
            # fully connected
            layers.Flatten(),                          # 5×5×16 = 400
            layers.Dense(120, activation="tanh"),
            layers.Dense(84, activation="tanh"),
            layers.Dense(num_classes, activation="softmax", dtype="float32"),
        ]
    )
    model.compile(
        optimizer="adam", 
        loss="sparse_categorical_crossentropy", 
        metrics=["accuracy"]
    )
    return model

# ─────────────────────────────────────────────────────────────────────────────
#  Deeper CNN for EMNIST Letters (26 classes)
# ─────────────────────────────────────────────────────────────────────────────
def build_deeper_cnn_letters(num_classes: int):
    """
    A deeper ConvNet that routinely hits 95%+ on EMNIST-Letters.
    Input in [0,1], shape=(28,28,1).
    """
    model = keras.Sequential(
        [
            layers.Input(shape=(28, 28, 1)),
            data_augmentation,
            layers.Conv2D(32, kernel_size=3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.Conv2D(32, kernel_size=3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),     # → 14×14×32
            layers.Dropout(0.25),

            layers.Conv2D(64, kernel_size=3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.Conv2D(64, kernel_size=3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),     # → 7×7×64
            layers.Dropout(0.25),

            layers.Conv2D(128, kernel_size=3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.Conv2D(128, kernel_size=3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),     # → 3×3×128
            layers.Dropout(0.25),

            layers.Flatten(),
            layers.Dense(512, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax", dtype="float32"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

# ─────────────────────────────────────────────────────────────────────────────
#  Preprocessing helpers for “predict” time
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_digit_image(uploaded_image) -> np.ndarray:
    """
    For digits: resize to 28×28 grayscale, normalize, no rotate/flip.
    Returns shape (1,28,28,1).
    """
    pil = Image.open(uploaded_image).convert("L").resize((28, 28))
    arr = np.array(pil).astype("float32")
    if arr.mean() > 127:
        arr = 255.0 - arr
    arr = arr / 255.0
    arr = np.expand_dims(arr, axis=-1)
    return np.expand_dims(arr, axis=0)  # (1,28,28,1)

def preprocess_letter_image(uploaded_image) -> np.ndarray:
    """
    For letters: rotate 90° CW + flip L↔R exactly as during EMNIST train.
    Returns shape (1,28,28,1), normalized to [0,1].
    """
    pil = Image.open(uploaded_image).convert("L").resize((28, 28))
    arr = np.array(pil).astype("float32")
    if arr.mean() > 127:
        arr = 255.0 - arr
    # now rotate + flip
    pil2 = Image.fromarray(arr.astype("uint8")).rotate(-90, expand=False)
    pil2 = pil2.transpose(Image.FLIP_LEFT_RIGHT)
    arr2 = np.array(pil2).astype("float32") / 255.0
    arr2 = np.expand_dims(arr2, axis=-1)
    return np.expand_dims(arr2, axis=0)  # (1,28,28,1)

# ─────────────────────────────────────────────────────────────────────────────
#  Streamlit UI
# ─────────────────────────────────────────────────────────────────────────────
st.title("LeNet-5 & Deep CNN Classifier: Digits (0–9) vs. Letters (A–Z)")
st.write("""
Choose either “Digits” or “Letters” in the sidebar, then:
- Train the corresponding model
- Upload a test image and get a prediction
""")

mode = st.sidebar.radio("Select mode", ["Digits", "Letters"])

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION A: TRAIN / EVALUATE DIGITS
# ─────────────────────────────────────────────────────────────────────────────
if mode == "Digits":
    st.header("▶️ Train / Evaluate Digit-Only Model (MNIST 0–9)")

    if st.button("Train Digit Model"):
        with st.spinner("Loading MNIST data..."):
            x_d, y_d = load_mnist_data()
        # shuffle + split
        idx = np.arange(len(x_d))
        np.random.shuffle(idx)
        x_d, y_d = x_d[idx], y_d[idx]
        split = int(0.9 * len(x_d))
        x_train, y_train = x_d[: split], y_d[: split]
        x_val, y_val = x_d[split:], y_d[split:]

        with st.spinner("Building & training LeNet-5 on MNIST..."):
            digit_model = build_lenet5(num_classes=10)
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor="val_accuracy", patience=5, restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
                ),
            ]
            digit_model.fit(
                x_train,
                y_train,
                epochs=5,
                batch_size=128,
                validation_data=(x_val, y_val),
                callbacks=callbacks,
                verbose=1,
            )

        os.makedirs("models", exist_ok=True)
        digit_model.save("models/lenet5_digits.h5")
        st.success("✅ Digit model trained and saved to `models/lenet5_digits.h5`")

    st.markdown("---")
    st.subheader("Upload a digit image to classify (0–9)")
    uploaded = st.file_uploader(
        "PNG/JPG that contains one handwritten digit (0–9)", type=["png", "jpg", "jpeg"]
    )
    if uploaded:
        st.image(uploaded, caption="Uploaded Image", use_column_width=True)
        img_arr = preprocess_digit_image(uploaded)

        if not os.path.exists("models/lenet5_digits.h5"):
            st.error("❌ Digit model not found. Train it first.")
            st.stop()

        model_d = load_model_from_path("models/lenet5_digits.h5")
        preds = model_d.predict(img_arr)
        digit_names = [str(i) for i in range(10)]
        pred_index = int(np.argmax(preds[0]))
        conf = float(np.max(preds[0]) * 100.0)
        st.write(f"**Predicted Digit:** {digit_names[pred_index]}")
        st.write(f"**Confidence:** {conf:.2f}%")
        st.write("**Top 3 Digit Predictions:**")
        top3 = np.argsort(preds[0])[::-1][:3]
        for i in top3:
            st.write(f"{digit_names[i]}: {preds[0][i]*100:.2f}%")
        st.write("**Digit Model Summary:**")
        model_d.summary(print_fn=st.write)

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION B: TRAIN / EVALUATE LETTERS
# ─────────────────────────────────────────────────────────────────────────────
else:  # mode == "Letters"
    st.header("▶️ Train / Evaluate Letter-Only Model (EMNIST A–Z)")

    if st.button("Train Letter Model"):
        with st.spinner("Loading EMNIST letters data..."):
            x_l, y_l = load_emnist_letters_data()
        # shuffle + split
        idx = np.arange(len(x_l))
        np.random.shuffle(idx)
        x_l, y_l = x_l[idx], y_l[idx]
        split = int(0.9 * len(x_l))
        x_train, y_train = x_l[: split], y_l[: split]
        x_val, y_val = x_l[split:], y_l[split:]

        with st.spinner("Building & training Deep CNN on EMNIST letters..."):
            letter_model = build_deeper_cnn_letters(num_classes=26)
            callbacks = [
                # Give it up to 10 epochs without val_accuracy improvement
                keras.callbacks.EarlyStopping(
                    monitor="val_accuracy", patience=10, restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
                ),
            ]
            letter_model.fit(
                x_train,
                y_train,
                epochs=5,
                batch_size=128,
                validation_data=(x_val, y_val),
                callbacks=callbacks,
                verbose=1,
            )

        os.makedirs("models", exist_ok=True)
        letter_model.save("models/deep_emnist_letters.h5")
        st.success("✅ Letter model trained and saved to `models/deep_emnist_letters.h5`")

    st.markdown("---")
    st.subheader("Upload a letter image to classify (A–Z)")
    uploaded = st.file_uploader(
        "PNG/JPG that contains one handwritten letter (A–Z)", type=["png", "jpg", "jpeg"]
    )
    if uploaded:
        st.image(uploaded, caption="Uploaded Image", use_column_width=True)
        img_arr = preprocess_letter_image(uploaded)

        if not os.path.exists("models/deep_emnist_letters.h5"):
            st.error("❌ Letter model not found. Train it first.")
            st.stop()

        model_l = load_model_from_path("models/deep_emnist_letters.h5")
        preds = model_l.predict(img_arr)
        # class_names: 0→A, 1→B, …, 25→Z
        letter_names = [chr(i) for i in range(65, 91)]
        pred_index = int(np.argmax(preds[0]))
        conf = float(np.max(preds[0]) * 100.0)
        st.write(f"**Predicted Letter:** {letter_names[pred_index]}")
        st.write(f"**Confidence:** {conf:.2f}%")
        st.write("**Top 3 Letter Predictions:**")
        top3 = np.argsort(preds[0])[::-1][:3]
        for i in top3:
            st.write(f"{letter_names[i]}: {preds[0][i]*100:.2f}%")
        st.write("**Letter Model Summary:**")
        model_l.summary(print_fn=st.write)
