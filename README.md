Blood group detection using fingerprint 

Project Overview

This project implements a non-invasive blood group detection system using Convolutional Neural Networks (CNN) .
The system analyzes fingerprint patterns and predicts the corresponding blood group using a trained deep learning model.
A web-based interface is developed using Flask to allow users to upload fingerprint images and obtain results through a browser.

Technologies Used :

Python
Convolutional Neural Networks (CNN)
TensorFlow and Keras
NumPy and OpenCV
Flask
HTML and CSS
Visual Studio Code

System Description
The user uploads a fingerprint image through the web interface.
The uploaded image is preprocessed and passed to the trained CNN model.
The model extracts fingerprint features and predicts the blood group.
The predicted result is displayed on the browser.

Steps to run the project
1.Clone the repository :
git clone https://github.com/Kushikc/Blood-Group-detection-using-CNN.git
cd Blood-Group-detection-using-CNN

2.(Optional) Create and activate a virtual environment :
python -m venv fingerprint_env
fingerprint_env\Scripts\activate

3.Install required libraries :
pip install -r requirements.txt

4.Run the Flask application :
python app.py

5.Open the browser and go to :
http://127.0.0.1:5000

