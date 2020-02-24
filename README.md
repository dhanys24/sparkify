# Sparkify Customer Churn Prediction - Capstone Project
Make a model to predict customer who will churn


### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#projectmotivation)
3. [File Descriptions](#filedescriptions)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensingauthorsandacknowledgements)

## Installation <a name="installation"></a>

We need to install software jupyter notebook (python 3.6) with several libraries installed :
- Pyspark
- Numpy
- Pandas
- Seaborn

## Project Motivation <a name="projectmotivation"></a>
This project will help Sparkify as song application company to predict customer who wil be churned. 
For company to retain existing customer is easier than acquiring new customer.Also in most cases it will be cost efficient to retain the existing one than new.

## File Descriptions <a name="filedescriptions"></a>
- sparkify_capstone_project.ipynb : Jupyter notebook used for this project
- sparkify_capstone_project.py: python script extracted from the notebook

## Result <a name="results"></a>
I use 12 features to predict users churn. Two of three model I test, Logistic Regression and Gradient Boosting Tree show that two most important features are Thumbsdown and Duration of User Subscribe the Sparkify. I try 3 methods for modeling that are : Logistic Regression, Random Forest Classifier, and Gradient Boosting Tree. Based on the F1-Score, Logistic Regression is the most fittable for this case with F1-Score 0.7976190476190476
Result can be found on Medium link [here](https://medium.com/@dhanys24/sparkify-project-predicting-customer-churn-e106f2c94729).

## Licensing, Authors, and Acknowledgements  <a name="licensingauthorsandacknowledgements"></a>
Thanks to [udacity](https://www.udacity.com/course/machine-learning-engineer-nanodegree--nd009t) for providing the data and jupyter notebook workspace for this Sparkify Project
