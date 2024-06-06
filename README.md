# Human-Emotion-Detector

## Overview
- A Human Emotion Detector app built using CNN and OpenCV. The webpage is developed using Flask.

## How to run this app:
- To run this application you need Python installed in your local machine.
- Download `haarcascade_frontalface_default.xml`, `app.py`, `requirements.txt` and the 'static' and 'templates' folder by cloning the repository.
- In your environment install all the libraries and dependencies by using `pip install -r requirements.txt`
- Then run the app directly by typing `python app.py` which will open the webpage and start displaying the detected emotions in real time.
- To close the application click the 'Exit' button.

## Description of the project
- The dataset is a subset of the famous FER-13 dataset of human emotions. It is taken from Kaggle.
- The model is a simple CNN model comprising of five Conv2D layers intersparsed with BatchNormalization layers to normalize the data and MaxPooling2D is used to pool the results.
- To prevent overfitting Dropout layers are also used when necessary.
- After Flattening the model, it is passed through a Dense layer. The final layer is also a Dense layer with 5 outputs and activation function being Softmax.
- The model is trained using an Adam optimizer and categorical_crossentropy loss for 15 epochs.
- The history of training is given in `plot.png` and the model is saved in `model.keras`.
- The webpage is made using Flask and the necessary HTML and CSS files are there in the static and templates folder. It uses OpenCV to read data from the webcam directly and give the response.
- HaarCascade is used to recognize the face and draw the bounding box.
