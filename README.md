# 🤟 Sign-to-Sentence: Real-Time Sign Language Recognition and Translation System

> 🎓 **Team Mini Project** | Foundation for our ongoing Sign Language Recognition major project

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Holistic-orange)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-FF6F00?logo=tensorflow)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Random%20Forest-F7931E?logo=scikitlearn)

---

## 📖 Overview

Sign-to-Sentence is a real-time sign language recognition system developed using **Computer Vision** and **Machine Learning** to help bridge the communication gap between hearing-impaired individuals and non-sign language users.

The system captures live webcam input, extracts **hand**, **pose**, and **facial landmarks** using **MediaPipe Holistic**, predicts sign gestures using a **Random Forest classifier**, and converts recognized signs into meaningful English sentences through a language mapping module.

This project serves as the **foundation for our ongoing major project**, where we are extending the system with continuous sign recognition, deep learning models, and improved natural language generation.

---

## 🎯 Problem Statement

Sign language is one of the primary communication methods for people with hearing and speech impairments. However, many individuals are not familiar with sign language, making everyday communication difficult.

This project aims to reduce this communication gap by recognizing sign language gestures in real time using computer vision and machine learning techniques. The system translates recognized gestures into meaningful English text without requiring wearable sensors or specialized hardware, making it an accessible and cost-effective solution.

## Objectives

- Develop a real-time sign language recognition system using artificial intelligence and computer vision.
- Extract important spatial and temporal features from sign language videos.
- Recognize continuous sign sequences using deep learning models.
- Convert recognized signs into meaningful sentences.
- Provide an accessible communication solution for individuals with hearing and speech impairments.

---

## Features

- Real-time sign language recognition using webcam input
- Vision-based approach without wearable sensors
- Hand, pose, and facial landmark extraction
- Spatiotemporal feature learning for dynamic gestures
- Deep learning-based sequence recognition
- Continuous sign sequence processing
- Text generation from recognized signs
- Scalable architecture for future speech and multilingual support
Video Input
|
↓
Frame Extraction
|
↓
Landmark Detection
(Hand, Pose, Face)
|
↓
Feature Extraction & Normalization
|
↓
Deep Learning Model
(CNN / LSTM Based Architecture)
|
↓
Sign Sequence Recognition
|
↓
Sentence Generation
|
↓
Text Output


---

## Technologies Used

### Programming Language
- Python

### Computer Vision
- OpenCV
- MediaPipe

### Deep Learning
- TensorFlow
- Keras
- LSTM Networks
- CNN-based Feature Extraction

### Data Processing
- NumPy
- Pandas
- Scikit-learn

### Development Tools
- Visual Studio Code
- Jupyter Notebook
- Git & GitHub

---
