# Assignment 2: Enhanced Neural Networks

## Official Dataset: https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008
## Official Paper: https://www.hindawi.com/journals/bmri/2014/781670/ 

Dataset: Diabetes 130-US hospitals for years 1999-2008 Data Set
<ul>
    <li>Data: Diabetic Encounters (1-14 days/each) from 130 Hospitals for 10 years (1999-2008) </li>
    <li>Goal: Predict if a diabetic patient will be readmitted to a hospital (less than 30 days, after 30 days, or never)</li>
    <li>Target Feature: readmitted </li>
</ul>

Used two free days

Preprocessing:
    Separte file for precossing. Only need to run the NeuralNet.py
    80/20 train/test split

Neural Net:
    Only one hidden layer for now and you can change the number of neurons in the layer 

Optimizer:
    Movementum added with deepcopy and movementum value of 0.9

Additional Notes:
 - We modified the starter code by creating a separate module for preprocessing
