# Hamlet Next Word Predictor (LSTM)

This project implements a next-word prediction model trained on the complete text of William Shakespeare's *Hamlet*. The model uses a Long Short-Term Memory (LSTM) neural network to learn patterns in the text and predict the most probable next word given a sequence of input words. A Streamlit application is provided for easy interaction with the trained model.

## Table of Contents

- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Details](#-model-details)
- [Contributing](#-contributing)
- [License](#-license)

## Features

- **Text Preprocessing**: Cleans and tokenizes the *Hamlet* text.
- **LSTM Model**: A deep learning model capable of capturing sequential dependencies in text.
- **Next Word Prediction**: Predicts the next word based on user input.
- **Streamlit Web Application**: An interactive and user-friendly interface for real-time predictions.

## ⚙Installation

To set up the project locally, follow these steps:

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/your-username/hamlet-next-word-predictor.git](https://github.com/your-username/hamlet-next-word-predictor.git)
    cd hamlet-next-word-predictor
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Download NLTK data (if not already downloaded):**
    The `experiments.ipynb` notebook will attempt to download the `gutenberg` corpus automatically. If you run into issues, you can download it manually:
    ```python
    import nltk
    nltk.download('gutenberg')
    ```

## Usage

### 1. Train the Model

The `models/` directory should contain `hamlet_lstm_model.keras` and `hamlet_tokenizer.pkl` after running the training notebook. If these files are missing or you wish to retrain the model, execute the `experiments.ipynb` notebook:

1.  Open the notebook:
    ```bash
    jupyter notebook experiments.ipynb
    ```
2.  Run all cells in the notebook. This will:
    * Download the *Hamlet* text.
    * Preprocess the text and create sequences.
    * Train the LSTM model.
    * Save the trained model and tokenizer to the `models/` directory.

### 2. Run the Streamlit Application

Once the model and tokenizer files are available, you can run the Streamlit application:

```bash
streamlit run app.py
```

## Model Details
The core of this project is an LSTM-based neural network.

### Architecture:

  * Embedding Layer: Converts word indices into dense vectors.
  * LSTM Layers: Two LSTM layers to capture long-range dependencies in the text.
  * Dropout Layers: Applied after LSTM layers to prevent overfitting.
  * Dense Layer: An output layer with a softmax activation function to predict the probability distribution over the vocabulary.
  * Training Data: The model is trained on the entire text of William Shakespeare's Hamlet, sourced from the NLTK Gutenberg corpus.
  * Loss Function: Categorical Crossentropy.
  * Optimizer: Adam.
  * Metrics: Accuracy.
