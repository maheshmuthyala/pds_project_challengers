# pds_project_challengers
In this project we used 4 datasets and 3 statistical Imputation methods and KNN Imputation method and 4 Machine Learning algorithms.

# Dataset
The datasets are collected from Kaggle and data world.
The names of the datasets are mammographic_masses.data, ParisHousingClass, riceClassification, and cancer patient data sets - data world.
All datasets are numeric datasets

# Imputation Methods
The statistical methods are imported from sklearn
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
From SimpleImputer we used mean, median, and mode imputation methods.
From KNNImputer we used the KNN imputation method

# Machine Learning Algorithms
We used 4 Machine Learning Algorithms
1. Logistic Regression
from sklearn.linear_model import LogisticRegression
2. Support vector Machine
from sklearn.multiclass import OneVsRestClassifier
3. Naive Bayes Classifier
from sklearn.naive_bayes import GaussianNB
4. Decision Tree Classification
from sklearn.tree import DecisionTreeClassifier

The above-mentioned libraries are imported and used for the project execution
The accuracy and Confusion matrix of every algorithm is produced

# Code Execution
We used Google Collab to run the code.
In the code we are importing the dataset from google drive the datasets are numeric in CSV file format we imported one dataset at a time and executed the code so we run the code 4 times to import 4 datasets.
Statistical imputation methods mean, median, and mode
impute='median'   # Can use these strategies: ['mean', 'median', 'most_frequent'] Change strategies and run code again
s_imputer=SimpleImputer(missing_values=np.nan,strategy=impute)  
we change the value to variable impute and run the code again to fill data in the dataset accordingly we change them after noting the accuracies of the ML algorithms
In the below line we need to change the values according to the dimensions of the dataset
s_imputer=s_imputer.fit(data.iloc[0:,1:11]) 
data.iloc[0:,1:11]=s_imputer.transform(data.iloc[0:,1:11]) 
we run code repeatedly just changing the imputation method and dataset and noting the accuracies of the ML algorithms

After noting the accuracies of the ML algorithms in different cases we plotted the graphs

# Results
We found that Mean and KNN Imputation methods are good for imputation to get good accuracy for ML algorithms and Decision tree classification is good for classification.
