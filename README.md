# Machine Learning Project: Heart Disease Prediction

## Authors
- Jhansi Bhaskarla
- Surendra Pothuri

## Table of Contents
1. Import Packages
2. Exploratory Data Analysis (EDA)
3. Preparing Machine Learning Models
4. Models Evaluation
5. Ensembling
6. Conclusion

## Project Overview
This project aims to predict heart disease using various machine learning algorithms. The dataset used in this project is the Heart Disease dataset, which contains multiple features related to patient health.

## Import Packages
The necessary packages for this project include:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- mlxtend

## Exploratory Data Analysis (EDA)
The initial data analysis involves:
1. Loading the dataset and displaying the first few rows.
2. Checking for null values and data types.
3. Descriptive statistics for numerical columns.
4. Visualizing distributions using histograms.
5. Visualizing correlations using a heatmap.

## Preparing Machine Learning Models
The machine learning models prepared and evaluated in this project include:
1. Logistic Regression
2. Naive Bayes
3. Random Forest Classifier
4. Extreme Gradient Boost
5. K-Nearest Neighbour
6. Decision Tree
7. Support Vector Machine

## Models Evaluation
Each model is evaluated based on the following metrics:
- Confusion Matrix
- Accuracy Score
- Precision, Recall, and F1-Score
- ROC Curve

## Ensembling
To improve the accuracy of predictions, an ensembling technique called stacking is used. The StackingCVClassifier from the mlxtend library is employed to combine multiple base classifiers.

## Conclusion
1. The Support Vector Machine (SVM) and the StackingCVClassifier demonstrated the highest accuracy, each achieving an impressive 98.05%. The Decision Tree model also performed notably well, with an accuracy of 94.63%.
2. Exercise-induced angina and chest pain are major symptoms of a heart attack.
3. The ensembling technique increased the accuracy of the model.

## Usage Instructions
1. Clone the repository.
2. Ensure all required packages are installed.
3. Load the dataset (heart.csv).
4. Run the Jupyter Notebook cells sequentially to reproduce the analysis and results.


