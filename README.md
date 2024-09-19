# Olympics-Prediction
The project aims for both the categories of Data Analytics and Machine Learning 

This project focuses on both analytics and machine learning-based predictions over Olympic events, teams, and nations. It provides a comprehensive analysis of Olympic data from various perspectives and implements predictive models using TensorFlow and Keras to forecast outcomes in Olympic events.

Table of Contents
•	Project Overview
•	Features
•	Installation
•	Usage
•	Model Architecture
•	Dataset
•	Technologies Used
•	Performance Metrics
•	Screenshots
•	License


Project Overview
The Olympics Prediction project combines data analytics and machine learning to provide insights and predictions about Olympic events, teams, and athletes. The analytical part covers nation-wise, event-wise, and athlete-wise analysis, while the prediction model forecasts future Olympic event outcomes based on historical data.

Features

Analytics:
•	Nation-wise analysis: Breakdown of medals won by different nations over the years.
•	Event-wise analysis: Insights into the popularity and outcomes of specific events.
•	Athlete-wise analysis: Performance trends of individual athletes.
•	Overall analysis: A complete overview of Olympic performances and trends.

Predictions:
•	Machine Learning Models: Uses TensorFlow and Keras for building predictive models.
•	Prediction Algorithms: Models based on activation functions like ReLU and Sigmoid.

Installation
1.	Clone the repository:
git clone https://github.com/username/olympics-prediction.git
2.	Navigate to the project directory:
cd olympics-prediction
3.	Install dependencies:
pip install -r requirements.txt

Model Architecture
The predictive model is a neural network designed with the following architecture:
•	Input Layer: Preprocessed Olympic data features.
•	Hidden Layers: Multiple dense layers with ReLU activation function.
•	Output Layer: Uses Sigmoid activation for binary classification (win/loss) prediction.

Key Technologies Used:
•	TensorFlow: For building and training the neural networks.
•	Matplotlib and Seaborn – Python Libraries used for Data Visualisation .
•	Keras: High-level neural network API for TensorFlow.
•	ReLU Activation: Used in the hidden layers for model optimization.
•	Sigmoid Activation: Used in the output layer for binary classification.

Dataset
•	The project uses historical Olympic datasets that include information about nations, events, athletes, and results.
•	Preprocessing: Missing values were handled, categorical features were one-hot encoded, and the data was normalized for better model performance.
https://www.kaggle.com/datasets/heesoo37/120-years-of-olympic-history-athletes-and-results


Performance Metrics

Metric	  Value
Accuracy	90.6%
AUC	      87.7%
Loss      26.7%





