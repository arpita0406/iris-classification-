## Overview
This project uses machine learning to classify iris flowers into three species (Setosa, Versicolor, Virginica) based on their physical measurements.

## Problem Statement
The Iris dataset contains measurements of 150 iris flowers from three different species. The challenge is to build a model that can accurately predict the species based on four features:
- Sepal length
- Sepal width
- Petal length
- Petal width

## Features
1. **Data Loading** - Loads the famous Iris dataset with 150 samples
2. **Data Splitting** - Splits data into training and testing sets
3. **KNN Classification** - Implements K-Nearest Neighbors algorithm
4. **Decision Tree Classification** - Implements Decision Tree algorithm
5. **Model Comparison** - Compares accuracy of both models
6. **Prediction** - Makes predictions on sample data

## Algorithms Used

### K-Nearest Neighbors (KNN)
Uses the 3 nearest data points to classify flowers. Simple and works well for this dataset.

### Decision Tree
Creates a tree structure for decision making by splitting data based on feature values.

## Technologies Used

- **Python** - Programming language
- **Scikit-learn** - Machine learning library
- **Pandas** - Data manipulation
- **NumPy** - Numerical operations

## How to Run

### Installation
```bash
pip install pandas scikit-learn numpy
```

## Output
The program displays accuracy scores for both models, confusion matrices showing classification performance, comparison of which model performed better, and a sample prediction demonstrating how to use the trained model.

## Application
This project demonstrates supervised learning classification and helps understand how different algorithms work on the same dataset.