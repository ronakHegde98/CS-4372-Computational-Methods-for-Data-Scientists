# CS 4372: Assignment 3 - Convolutional Neural Networks and Transfer Learning

Authors: Ronak Hegde and Vignesh Vasan

Goal: Image Classification using CNN (Transfer Learning + Data Augmentation) in Keras
 
## Official dataset: https://data.mendeley.com/datasets/4drtyfjtfy/1

Steps to build project:
1. Run the bulid_dataset.py to pull the dataset from the internet
2. It will make a folder called data with another folder called weather_dataset with a test, validation, and train split folders. 
3. Run the CNN notebook to download tensorflow (we used tensorflow version 2.3) to run the CNN (pre-trained; Inception Model)
4. results.txt contains 25 picture paths with associated true label and predicted label
5. experiments.log contains varying experiments in hyperparameter tuning of our model