# CNN Bias Demonstration

This Streamlit application demonstrates how a Convolutional Neural Network (CNN) can be trained on different datasets, showcasing bias when the network is exposed to different types of data. It allows users to:

- Select a dataset to train the CNN model (Numbers, Shapes, or Alphabets).
- Train the model on the selected dataset.
- Upload their own image for classification, and view the model's predictions.

## Requirements

numpy
pillow
tensorflow==2.19.0
  # or latest stable, CPU version
tensorflow-datasets
streamlit

## Activate the Environment

```bash
.\cnn_bias_demo\Scripts\activate
```

### Install Dependencies

You can install the required dependencies using the following command:

```bash
pip3 install -r requirements.txt
```
## How to Run

### Run the Streamlit app

After installing the required dependencies, you can run the app using Streamlit by executing:

```bash
python -m streamlit run cnn_bias_demo.py  
```
Open your browser and go to http://localhost:8501 to use the application.



This part allows you to run the Streamlit app after setting up the dependencies and provides the link to access the application in the browser.

## Functionality

### Datasets

You can choose from three datasets to train the CNN:

- **Numbers**: Uses the MNIST dataset (digits 0-9).

### Training the Model

- The model is trained using **TensorFlow** and **Keras**.
- Data augmentation techniques (rotation, zoom, translation, flip) are applied to increase robustness during training.
- Upon selecting a dataset and clicking "Train Model", the model is trained on the chosen dataset. Once training is complete, the model is saved for future use.

### Image Classification

- Users can upload an image for classification.
- The image is preprocessed, normalized, and passed through the trained CNN.
- The predicted class and confidence score will be displayed, along with the top 3 predictions.

---

## Model Architecture

The CNN model consists of the following layers:

- **Input Layer**: Shape `(28, 28, 1)` for grayscale images.
- **Data Augmentation**: Random rotation, zoom, translation, and flip to make the model more robust.
- **Convolutional Layers**:
  - 64 filters with a kernel size of `(3,3)`
  - 128 filters with a kernel size of `(3,3)`
  - 128 filters with a kernel size of `(3,3)`
- **MaxPooling Layers**: Pooling layer after each convolutional layer.
- **Flatten Layer**: Flatten the 2D feature maps into a 1D vector.
- **Dense Layers**:
  - 128 units with ReLU activation
  - Output layer with a softmax activation function for multi-class classification.

---

## Model Evaluation

The model's performance is evaluated using **accuracy** and **validation loss** during training. A "success" message is shown when training is completed. If a model is already trained, it can be loaded for predictions on new images.


## File Structure

```plaintext
.
├── app.py               # Main Streamlit app script
├── models/              # Directory to store trained models
│   ├── cnn_numbers.h5   # Saved model for 'Numbers' dataset
│   ├── cnn_shapes.h5    # Saved model for 'Shapes' dataset
│   └── cnn_alphabets.h5 # Saved model for 'Alphabets' dataset
└── requirements.txt     # File containing the required Python packages

```
---

## License

This project is licensed under the **MIT License** - see the LICENSE file for details.



