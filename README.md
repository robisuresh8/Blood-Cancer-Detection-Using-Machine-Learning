# Blood Cancer Detection Using Deep Learning

This project is a web application that uses a deep learning model to detect blood cancer (specifically, Leukemia) from microscopic blood cell images.

The application is built with a **Flask** backend, which serves a pre-trained **Keras (TensorFlow)** model. Users can upload an image of a blood cell, and the model will classify it as cancerous or non-cancerous.

## Features

* **Deep Learning Model:** Utilizes a Convolutional Neural Network (CNN), likely built using transfer learning, to classify images.
* **Web Interface:** A simple, user-friendly web app built with Flask for uploading images and viewing predictions.
* **Pre-trained Model:** Includes a pre-trained Keras model (`keras_model2.h5`) for immediate use.

## Technology Stack

* **Backend:** Python, Flask
* **Deep Learning:** TensorFlow, Keras
* **Image Processing:** Pillow (PIL), NumPy
* **Frontend:** HTML (served by Flask Templates)

## How to Run This Project

### 1. Clone the Repository

```bash
git clone [https://github.com/robisuresh8/Blood-Cancer-Detection-Using-Machine-Learning.git](https://github.com/robisuresh8/Blood-Cancer-Detection-Using-Machine-Learning.git)
cd Blood-Cancer-Detection-Using-Machine-Learning
```

### 2. Create a Virtual Environment

It's highly recommended to use a virtual environment to manage dependencies.

```bash
# Create a virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

This project does not include a `requirements.txt` file, but you can install the necessary libraries using pip.

```bash
pip install Flask tensorflow numpy pillow
```

### 4. Run the Flask Application

Once the dependencies are installed, you can start the Flask server.

```bash
python mySite.py
```

### 5. View in Browser

Open your web browser and navigate to the following address:

`http://127.0.0.1:5000`

You should now be able to upload a blood cell image and receive a prediction.

## Project Structure

```
.
├── Data/                   # Folder for the training/testing dataset
├── static/                 # Static files (CSS, JS) for the web app
├── templates/              # HTML templates for the Flask app
├── tfenv/                  # (Virtual environment - should be in .gitignore)
├── upload/                 # Directory for user-uploaded images
├── keras_model2.h5         # The pre-trained Keras deep learning model
├── mySite.py               # The main Flask application file
├── supportFile.py          # Helper functions for the application
├── transferLearning.py     # The script used to train the Keras model
└── ...
```

## How to Re-train the Model

The script `transferLearning.py` appears to be the code used to train the model. To run it, you would typically:
1.  Ensure you have the full `Data/` dataset (images sorted into classes).
2.  Install the necessary libraries (like `tensorflow`, `numpy`).
3.  Run the script: `python transferLearning.py`.
4.  This will (re)generate the `keras_model2.h5` file.

---
### **Recommendation for you:**

To make it even easier for others, I recommend you create a `requirements.txt` file. You can do this easily:
1.  Activate your virtual environment (where you installed Flask, TensorFlow, etc.).
2.  Run this command in your terminal:
    ```bash
    pip freeze > requirements.txt
    ```
3.  This will create a `requirements.txt` file. Add, commit, and push this file to your GitHub repository!
