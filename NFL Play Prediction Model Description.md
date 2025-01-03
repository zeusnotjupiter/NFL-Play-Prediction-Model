NFL Play Prediction Model
This project implements a machine learning model to predict whether an NFL play will be a pass or run based on game situation factors. The model uses Random Forest Classification to analyze features like down, distance, field position, time remaining, and offensive formation to make its predictions.
Features

Real-time play prediction based on current game situation
Advanced analysis mode with detailed feature input
Model performance tracking with accuracy and ROC AUC metrics
Continuous learning capability through model updates with actual play outcomes
Support for various offensive formations (Shotgun, Under Center, Pistol, No Huddle)
Interactive command-line interface for easy use

Technical Implementation

Machine Learning Model: Random Forest Classifier with feature selection
Data Processing: StandardScaler for feature normalization
Feature Engineering: Includes formation one-hot encoding and time conversion
Model Persistence: Save/load functionality using joblib
Error Handling: Comprehensive logging and input validation

Requirements
pythonCopypandas
numpy
scikit-learn
joblib
Setup and Installation

Clone the repository:

bashCopygit clone https://github.com/[your-username]/NFL-Play-Prediction-Model.git
cd NFL-Play-Prediction-Model

Install required packages:

bashCopypip install pandas numpy scikit-learn joblib

Ensure you have the required data file:


Place plays.csv in the project directory

Usage
Run the main script:
bashCopypython nfl_prediction_ai.py
The program offers two modes:

Quick Prediction Mode: Fast input of basic game situation details
Detailed Analysis Mode: Comprehensive feature input with model updating

Input Features:

Down (1-4)
Yards to go
Field position
Quarter
Time remaining
Offensive formation

Model Performance
The model's performance is evaluated using:

Accuracy score
ROC AUC score
Classification report with precision, recall, and F1-score

Future Improvements

Web interface for easier interaction
Integration with live game data
Additional features (weather, personnel packages, etc.)
Support for more complex play type predictions

Contributing
Feel free to fork the repository and submit pull requests with improvements.
License
This project is licensed under the MIT License - see the LICENSE file for details.