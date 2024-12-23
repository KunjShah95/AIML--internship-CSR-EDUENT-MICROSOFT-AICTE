# SMS Spam Detection Using NLP 📱✉️

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/streamlit-0.84.0-brightgreen)

## Table of Contents

- [Introduction](#introduction)
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Streamlit App](#streamlit-app)
- [Contributors](#contributors)
- [License](#license)

## Introduction

This project focuses on detecting spam messages using Natural Language Processing (NLP) techniques. The goal is to build a model that can accurately classify SMS messages as spam or ham (non-spam).

## Project Overview

The project involves the following steps:

1. **Data Collection**
2. **Data Preprocessing**
3. **Feature Extraction**
4. **Model Training**
5. **Model Evaluation**
6. **Deployment**

## Dataset

The dataset used for this project is a collection of SMS messages labeled as spam or ham. The dataset can be found [here](https://www.kaggle.com/uciml/sms-spam-collection-dataset).

## Installation

To get started with the project, clone the repository and install the required dependencies:

```bash
git clone https://github.com/KunjShah95/sms-spam-detection.git
cd sms-spam-detection
pip install -r requirements.txt
```

## Usage

To use the project, follow these steps:

1. **Data Preprocessing**: Preprocess the dataset by converting all text to lowercase and removing special characters.
2. **Feature Extraction**: Extract features from the preprocessed data using techniques such as bag-of-words or TF-IDF.
3. **Model Training**: Train a machine learning model using the extracted features and the labeled data.
4. **Model Evaluation**: Evaluate the performance of the trained model using metrics such as accuracy, precision, recall, and F1-score.
5. **Deployment**: Deploy the trained model to a production environment for real-time SMS spam detection.

## Model Training

The model is trained using various machine learning algorithms such as Naive Bayes, Support Vector Machines, and Random Forest. The training process involves:

1. **Preprocessing the text data** (tokenization, stemming, etc.)
2. **Extracting features** using techniques like TF-IDF
3. **Training the model** on the preprocessed data

## Evaluation

The model is evaluated using metrics such as accuracy, precision, recall, and F1-score. Cross-validation is also performed to ensure the model's robustness.

## Results

The final model achieved an accuracy of **XX%** on the test dataset. Detailed evaluation metrics are as follows:

- **Precision**: XX%
- **Recall**: XX%
- **F1-Score**: XX%

## Streamlit App

To run the Streamlit app for real-time SMS spam detection, follow these steps:

1. Ensure all dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
3. Open your web browser and go to `http://localhost:8501` to interact with the app.

## Contributors

- [KunjShah95](https://github.com/KunjShah95)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
